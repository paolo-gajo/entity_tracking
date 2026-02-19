import torch

def gather_losses(args,
                causal_lm_loss_fn,
                kl_loss_fn,
                # linear_refinement_loss_fn,
                max_margin_loss_fn,
                logits,
                batch,
                device,
                lhs):
    if args.use_causal_lm_loss:
        causal_lm_loss = causal_lm_loss_fn(logits, batch['input_ids'], batch['attention_mask'])
    else:
        causal_lm_loss = torch.tensor(0.0, device=device)
    if args.use_kl:
        kl_loss = kl_loss_fn(logits, batch['input_ids'], batch['attention_mask'])
    else:
        kl_loss = torch.tensor(0.0, device=device)
    if args.use_max_margin_loss:
        max_margin_loss = max_margin_loss_fn(lhs, batch['step_indices'], batch['binary_label'])
    else:
        max_margin_loss = torch.tensor(0.0, device=device)
    num_losses = (int(args.use_causal_lm_loss) + int(args.use_kl)
                #   + int(args.use_order_loss)
                  + int(args.use_max_margin_loss))
    total_loss = (causal_lm_loss +
                    kl_loss +
                    max_margin_loss
                    ) / num_losses
    return {
        'causal_lm_loss': causal_lm_loss,
        'kl_loss': kl_loss,
        'max_margin_loss': max_margin_loss,
        'total_loss': total_loss,
        }

class KLDivergenceLoss(torch.nn.Module):
    def __init__(self, ref_model):
        """
        Computes KL(Student || Reference).
        Args:
            ref_model: The pretrained reference model (frozen).
        """
        super().__init__()
        self.ref_model = ref_model
        self.ref_model.eval()
        # Ensure reference model is frozen
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def forward(self, student_logits, input_ids, attention_mask):
        """
        Args:
            student_logits: Logits from the model being trained [Batch, Seq, Vocab]
            input_ids: Input IDs [Batch, Seq]
            attention_mask: Attention Mask [Batch, Seq]
        """
        # Get Reference Logits
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        # Shift logits and masks (predict next token)
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()

        # KL Div requires log_target=False (default): input is log_probs, target is probs
        log_probs_student = torch.functional.F.log_softmax(shift_student_logits, dim=-1)
        probs_ref = torch.functional.F.softmax(shift_ref_logits, dim=-1)

        # Compute KL per token
        # reduction='none' returns [Batch, Seq_Len]
        kl_per_token = torch.functional.F.kl_div(log_probs_student, probs_ref, reduction='none').sum(dim=-1)
        
        # Mask and average
        num_valid_tokens = shift_mask.sum()
        
        if num_valid_tokens > 0:
            return (kl_per_token * shift_mask).sum() / num_valid_tokens
        return torch.tensor(0.0, device=input_ids.device)

class CausalLMLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, input_ids, attention_mask):
        # 1. Create Labels from Input IDs
        labels = input_ids.clone()
        # Mask padding tokens
        labels[attention_mask == 0] = self.ignore_index

        # 2. Shift Logits and Labels
        # Remove last logit, remove first label
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 3. Compute Loss
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

class MaxMarginLoss(torch.nn.Module):
    def __init__(self, alpha, activations):
        super().__init__()
        self.alpha = alpha
        self.activations = activations
    
    def forward(self, inputs, step_ids, binary_labels):
        if self.activations == 'non-negative':
            inputs = torch.abs(inputs)
        
        inputs_true = inputs[binary_labels == 1, ...]
        step_ids_true = step_ids[binary_labels == 1, ...]
        
        inputs_corrupted = inputs[binary_labels == 0, ...]
        step_ids_corrupted = step_ids[binary_labels == 0, ...]

        total_loss = torch.tensor(0.0, device=inputs.device)
        num_samples = 0
        
        positives = self.get_step_embeddings(inputs_true, step_ids_true) 
        negatives = self.get_step_embeddings(inputs_corrupted, step_ids_corrupted)

        """
        From `ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE`, Vendrov et al. 2016
        x \preceq y iff  x_i \geq y_i --> x entails y iff all coordinates of x are bigger than y's
        So from y we construct an orthant, or cone
        and all points that entail y are to be found within that orthant

        In the case of a culinary recipe, the sink entails the preceding steps
        up to the leaves, which are entailed by the paths from ingredient to sink.
        The leaves do not entail anything and are only entailed.
        The sink only entails, and is not entailed by anything.
        This means we want the sink to be the biggest embedding
        and the leaves to be the smallest.

        In terms of the original paper's terms, big means abstract, small means specific.
        In our terms, big entails small.
        
        However, we do not want the model to rely solely
        on the depth at which a step is found to determine its magnitude,
        otherwise random permutations and topological orders
        are gonna have the same similarity matrices.

        E(x, y) = || max(0, y - x)||^2
        ∑(u,v)∈P  E(f(u), f(v)) +  ∑(u',v')∈N  max{0, alpha - E(f(u'), f(v'))}
        
        """
        for H in positives:
            H_prev = H[:-1, :]
            H_next = H[1:, :] 
            
            '''
            For the positives, we want the later embeddings to be larger.
            We want the energy for (H_prev, H_next) pairs to be 0.
            This forces the next embedding to be at least as big as the previous.
            '''

            loss_pos = self.E(H_next, H_prev).mean()

            total_loss = total_loss + loss_pos
            num_samples += 1

        for H in negatives:
            H_prev = H[:-1, :] 
            H_next = H[1:, :] 
            '''
            For the negatives, we want the energy for (H_prev, H_next) pairs
            to be large, i.e. we want H_prev to be bigger than H_next by an alpha.
            '''
            loss_neg = torch.relu(self.alpha - self.E(H_next, H_prev)).mean()
            
            total_loss = total_loss + loss_neg
            num_samples += 1

        return total_loss / (num_samples + 1e-9)

    # E(x,y) = ||relu(y-x)||^2
    @staticmethod
    def E(x, y):
        return torch.relu(y - x).pow(2).sum(dim=-1)

    def get_step_embeddings(self,
                            hidden_states: torch.Tensor, # [batch, tokens, dim]
                            step_ids: torch.Tensor, # [batch, tokens]
                            ):
        # NOTE: step_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, ..., N, N, N]
        # This is true both for positives and negatives,
        # the difference is that in the positives the 1s
        # contain the actual tokens of the first step,
        # but in the shuffled negatives the tokens which are assigned 1s
        # actually would correspond to another step index.
        
        # Index 0 is reserved for non-content,
        # such as special tokens like EOS that we do not want to pool.

        H_list = []
        for b in range(hidden_states.size(0)):
            # Get the unique steps in order (e.g., [1, 2, 3, 4, 5])
            # We assume the target is the sorted recipe, so steps are monotonic
            step_order = step_ids[b].unique()
            step_order = step_order[step_order != 0] # Remove padding
            step_order = sorted(step_order.tolist())

            if len(step_order) < 2: continue
                
            # Pool embeddings for each step
            h_list = []
            for s in step_order:
                mask = (step_ids[b] == s)
                # Mean pool the tokens for this step
                h_list.append(hidden_states[b][mask].mean(dim=0))
                
            H = torch.stack(h_list) # [step_order, dim]
            H_list.append(H)
        return H_list # [batch, step_order, dim]
