# src/loss_functions.py
import torch        

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def gather_losses(args,
                causal_lm_loss_fn,
                kl_loss_fn,
                max_margin_loss_fn,
                logits,
                batch,
                device,
                lhs_mml,
                pos_loss=None,
                ):
    if args.use_clm: assert logits is not None
    if args.use_kl: assert logits is not None

    if args.use_clm:
        causal_lm_loss = causal_lm_loss_fn(logits, batch['input_ids'], batch['loss_mask'])
    else:
        causal_lm_loss = torch.tensor(0.0, device=device)

    if args.use_kl:
        kl_loss = kl_loss_fn(logits, batch['input_ids'], batch['loss_mask'])
    else:
        kl_loss = torch.tensor(0.0, device=device)

    if args.use_mml:
        max_margin_loss = max_margin_loss_fn(lhs_mml, batch['step_indices_mml'], batch['binary_label'])
    else:
        max_margin_loss = torch.tensor(0.0, device=device)

    pos_loss = pos_loss if pos_loss is not None else torch.tensor(0.0, device=device)

    total_loss = (
        causal_lm_loss * args.clm_lambda +
        kl_loss        * args.kl_lambda +
        max_margin_loss* args.mml_lambda +
        pos_loss       * args.pos_lambda
    )

    return {
        'causal_lm_loss':  causal_lm_loss,
        'kl_loss':         kl_loss,
        'max_margin_loss': max_margin_loss,
        'pos_loss':        pos_loss,
        'total_loss':      total_loss,
    }

class MaxMarginLoss(torch.nn.Module):
    """
    Computes the max-margin order embedding loss for directed acyclic graph (DAG) topologies.

    Reference:
        Vendrov et al., 2016. "Order-Embeddings of Images and Language".
        The original formulation enforces a partial order such that $x \preceq y \iff x_i \ge y_i \forall i$.
        In Vendrov et al., this geometry maps semantic hierarchies (hypernymy/hyponymy), where
        larger embedding vectors represent abstract concepts that entail specific concepts.

    Topological Adaptation:
        This implementation adapts the cone-inclusion geometry to model causal reachability in procedural DAGs.
        - Leaves (initial steps/ingredients) are mapped to the smallest magnitude embeddings.
        - Sinks (final steps) are mapped to the largest magnitude embeddings.
        - Transitive reachability is modeled as entailment: $H_{\text{sink}} \succeq H_{\text{leaf}}$.
        
    Divergence from Vendrov et al.:
        Vendrov et al. construct negative samples by drawing independent entities (e.g., mismatched image-caption pairs).
        This implementation constructs negatives via stochastic permutations of a single causal sequence.
        Consequently, random permutations may contain valid transitive edges. This necessitates the dynamic
        boolean masking implemented in the forward pass to prevent the injection of false-negative gradients 
        on structurally valid sub-paths.

    Objective function:
        $$ \mathcal{L} = \sum_{(u,v) \in \mathcal{P}} E(f(u), f(v)) + \sum_{(u',v') \in \mathcal{N}} \max(0, \alpha - E(f(u'), f(v'))) $$
        where $E(x, y) = \lVert \max(0, y - x) \rVert^2$.
    """
    def __init__(self, alpha, activations):
        super().__init__()
        self.alpha = alpha
        self.activations = activations

    @staticmethod
    def E(x, y):
        """
        Computes the asymmetric distance metric $E(x, y)$.s
        If $x \succeq y$ coordinate-wise, the distance evaluates to strictly zero.
        """
        return torch.relu(y - x).pow(2).mean(dim=-1)
        # return torch.relu(y - x).pow(2).sum(dim=-1)

    def forward(self, inputs, step_ids, binary_labels):
        """
        Computes the batched max-margin loss.

        Args:
            inputs: Tensor of token-level hidden states.
            step_ids: Tensor mapping tokens to discrete step indices.
            binary_labels: Tensor indicating whether the sequence is topologically valid (1) or a permutation (0).

        Returns:
            Scalar loss normalized by the number of valid pairwise evaluations.
        """
        if self.activations == 'non-negative':
            inputs = torch.abs(inputs)
        
        inputs_true = inputs[binary_labels == 1, ...]
        step_ids_true = step_ids[binary_labels == 1, ...]
        
        inputs_corrupted = inputs[binary_labels == 0, ...]
        step_ids_corrupted = step_ids[binary_labels == 0, ...]

        total_loss = torch.tensor(0.0, device=inputs.device)
        num_samples = 0
        
        positives, _ = self.get_step_embeddings(inputs_true, step_ids_true) 
        negatives, neg_orders = self.get_step_embeddings(inputs_corrupted, step_ids_corrupted)

        # Process ground-truth topological sequences
        for H in positives:
            H_prev = H[:-1, :]
            H_next = H[1:, :] 

            # Enforce E(H_next, H_prev) = 0, mapping H_next inside the cone of H_prev
            loss_pos = self.E(H_next, H_prev).mean()
            total_loss = total_loss + loss_pos
            num_samples += 1

        # Process stochastic permutations
        for H, order in zip(negatives, neg_orders):
            order_tensor = torch.tensor(order, device=H.device)
            
            # Isolate invalid topological transitions (u > v)
            # This masks out randomly preserved transitive edges to prevent optimization pathologies
            mask_invalid_topology = order_tensor[:-1] > order_tensor[1:]
            
            if mask_invalid_topology.any():
                H_prev = H[:-1, :][mask_invalid_topology]
                H_next = H[1:, :][mask_invalid_topology] 
                
                # Enforce margin \alpha for invalid transitions
                loss_neg = torch.relu(self.alpha - self.E(H_next, H_prev)).mean()
                total_loss = total_loss + loss_neg
                num_samples += 1

        return total_loss / (num_samples + 1e-9)

    def get_step_embeddings(self, hidden_states, step_ids):
        """
        Extracts pooled latent representations and discrete sequence orders from token-level tensors.

        Args:
            hidden_states: Tensor of shape [batch, seq_len, dim].
            step_ids: Tensor of shape [batch, seq_len] containing step indices.

        Returns:
            H_list: List of pooled step embedding tensors.
            order_list: List of corresponding discrete topological indices.
        """
        H_list = []
        order_list = []
        for b in range(hidden_states.size(0)):
            step_order_tensor = step_ids[b]
            step_order = step_order_tensor[step_order_tensor != 0]
            
            # Extract sequence order while preserving the permutation
            seen = set()
            sequential_order = [x.item() for x in step_order if not (x.item() in seen or seen.add(x.item()))]

            if len(sequential_order) < 2: continue
                
            h_list = []
            for s in sequential_order:
                mask = (step_ids[b] == s)
                h_list.append(hidden_states[b][mask].mean(dim=0))
                
            H = torch.stack(h_list)
            H_list.append(H)
            order_list.append(sequential_order)
            
        return H_list, order_list
    
class CausalLMLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, input_ids, attention_mask):
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


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