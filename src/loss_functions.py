import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Loss aggregation
# ---------------------------------------------------------------------------

def gather_losses(args,
                  causal_lm_loss_fn,
                  kl_loss_fn,
                  max_margin_loss_fn,
                  logits,
                  batch,
                  device,
                  last_hidden_state,
                  pos_loss=None,
                  stp_loss=None,
                  cos_loss_fn=None,
                  ):
    if args.use_clm:
        assert logits is not None
    if args.use_kl:
        assert logits is not None
    use_pooled_clm = (args.use_clm and args.pool_clm)
    assert not (use_pooled_clm and args.use_mml), "can't use both pooled clm and mml right now because only one step_indices route exists for both"
    if args.use_clm:
        causal_lm_loss = causal_lm_loss_fn(logits, batch['input_ids'], batch['clm_mask'])
    else:
        causal_lm_loss = torch.tensor(0.0, device=device)

    if args.use_kl:
        kl_loss = kl_loss_fn(logits, batch['input_ids'], batch['clm_mask'])
    else:
        kl_loss = torch.tensor(0.0, device=device)

    if args.use_mml:
        max_margin_loss = max_margin_loss_fn(last_hidden_state, batch['step_indices'], batch['binary_label'])
    else:
        max_margin_loss = torch.tensor(0.0, device=device)

    if args.use_cos and cos_loss_fn is not None:
        cos_loss = cos_loss_fn(last_hidden_state, batch['step_indices'], batch['binary_label'])
    else:
        cos_loss = torch.tensor(0.0, device=device)

    pos_loss = pos_loss if pos_loss is not None else torch.tensor(0.0, device=device)
    stp_loss = stp_loss if stp_loss is not None else torch.tensor(0.0, device=device)

    total_loss = (
        causal_lm_loss  * args.clm_lambda +
        kl_loss         * args.kl_lambda +
        max_margin_loss * args.mml_lambda +
        cos_loss        * args.cos_lambda +
        pos_loss        * args.pos_lambda +
        stp_loss        * args.stp_lambda
    )
    return {
        'causal_lm_loss':  causal_lm_loss,
        'kl_loss':         kl_loss,
        'max_margin_loss': max_margin_loss,
        'cos_loss':        cos_loss,
        'pos_loss':        pos_loss,
        'stp_loss':        stp_loss,
        'total_loss':      total_loss,
    }

class CausalLMLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, input_ids, clm_mask):
        labels = input_ids.clone()
        labels[clm_mask == 0] = self.ignore_index

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# ---------------------------------------------------------------------------
# 3.  Step Token Prediction Loss  (Section 3.2 of the paper)
#
#     The completion consists of real step-token vocabulary entries in the
#     correct topological order.  The loss is standard causal LM
#     cross-entropy evaluated only at the step-token positions (controlled
#     by stp_mask).
# ---------------------------------------------------------------------------

class StepTokenLoss(nn.Module):
    """
    CLM loss restricted to step-token positions.

    This is functionally identical to ``CausalLMLoss`` but kept as a
    separate class so the training loop can weight it independently via
    ``stp_lambda``.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, input_ids, stp_mask):
        """
        Args:
            logits:    [B, T, V] — full model logits
            input_ids: [B, T]    — input token ids (including step tokens)
            stp_mask: [B, T]    — 1 at step-token completion positions
        Returns:
            Scalar CE loss (averaged over valid positions).
        """
        labels = input_ids.clone()
        labels[stp_mask == 0] = self.ignore_index

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


# ---------------------------------------------------------------------------
# Shared helper: pool hidden states by step id
# ---------------------------------------------------------------------------

def _get_step_embeddings(hidden_states, step_ids):
    """
    For each sample in the batch, pool token-level hidden states into one
    vector per step and return them in sequential order.

    Returns:
        H_list    : list of [S, D] tensors  (one per sample with ≥2 steps)
        order_list: list of lists of ints   (step ids in observed order)
    """
    H_list = []
    order_list = []
    for b in range(hidden_states.size(0)):
        step_order_tensor = step_ids[b]
        step_order = step_order_tensor[step_order_tensor != 0]

        seen = set()
        sequential_order = [
            x.item() for x in step_order
            if not (x.item() in seen or seen.add(x.item()))
        ]

        if len(sequential_order) < 2:
            continue

        h_list = []
        for s in sequential_order:
            mask = (step_ids[b] == s)
            h_list.append(hidden_states[b][mask].mean(dim=0))

        H_list.append(torch.stack(h_list))
        order_list.append(sequential_order)

    return H_list, order_list


# ---------------------------------------------------------------------------
# 4.  Max-Margin Loss  — unchanged from your original
# ---------------------------------------------------------------------------

class MaxMarginLoss(nn.Module):
    """
    Computes the max-margin order embedding loss for directed acyclic graph (DAG) topologies.
    When proj_dim > 0, the loss operates on a learned projection of the hidden
    states, leaving the main representation unconstrained.
    """
    def __init__(self, alpha, activations, hidden_dim=0, proj_dim=0):
        super().__init__()
        self.alpha = alpha
        self.activations = activations
        self.proj = None
        if proj_dim > 0 and hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim),
                nn.ReLU(),
            )

    @staticmethod
    def E(x, y):
        return torch.relu(y - x).pow(2).mean(dim=-1)

    def forward(self, inputs, step_ids, binary_labels):
        if self.proj is not None:
            inputs = self.proj(inputs.to(self.proj[0].weight.dtype))
        elif self.activations == 'non-negative':
            inputs = torch.abs(inputs)
        inputs_true = inputs[binary_labels == 1, ...]
        step_ids_true = step_ids[binary_labels == 1, ...]

        inputs_corrupted = inputs[binary_labels == 0, ...]
        step_ids_corrupted = step_ids[binary_labels == 0, ...]
        
        total_loss = torch.tensor(0.0, device=inputs.device)
        num_samples = 0

        positives, _ = self.get_step_embeddings(inputs_true, step_ids_true)
        negatives, neg_orders = self.get_step_embeddings(inputs_corrupted, step_ids_corrupted)

        for H in positives:
            H_prev = H[:-1, :]
            H_next = H[1:, :]
            loss_pos = self.E(H_next, H_prev).mean()
            total_loss = total_loss + loss_pos
            num_samples += 1

        for H, order in zip(negatives, neg_orders):
            order_tensor = torch.tensor(order, device=H.device)
            mask_invalid_topology = order_tensor[:-1] > order_tensor[1:]
            if mask_invalid_topology.any():
                H_prev = H[:-1, :][mask_invalid_topology]
                H_next = H[1:, :][mask_invalid_topology]
                loss_neg = torch.relu(self.alpha - self.E(H_next, H_prev)).mean()
                total_loss = total_loss + loss_neg
                num_samples += 1

        return total_loss / (num_samples + 1e-9)

    def get_step_embeddings(self, hidden_states, step_ids):
        return _get_step_embeddings(hidden_states, step_ids)


# ---------------------------------------------------------------------------
# 5.  Cosine Contrastive Loss
#
#     Non-directional: only cares whether two step embeddings should be
#     similar (positive pair) or dissimilar (negative pair).
#
#     Positive pairs  — consecutive steps in a correctly-ordered sequence:
#         L_pos = 1 - cos_sim(h_i, h_{i+1})
#
#     Negative pairs  — consecutive steps where the ordering has been
#         corrupted (step_id[i] > step_id[i+1]):
#         L_neg = max(0, cos_sim(h_i, h_{i+1}) - alpha)
#
#     Total loss:
#         L = mean over positives of L_pos  +  mean over negatives of L_neg
# ---------------------------------------------------------------------------

class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss based on cosine similarity between pooled step embeddings.
    Non-directional: E(h_i, h_{i+1}) == E(h_{i+1}, h_i).
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs, step_ids, binary_labels):
        inputs_true      = inputs[binary_labels == 1]
        step_ids_true    = step_ids[binary_labels == 1]
        inputs_corrupted = inputs[binary_labels == 0]
        step_ids_corrupt = step_ids[binary_labels == 0]

        total_loss  = torch.tensor(0.0, device=inputs.device)
        num_samples = 0

        positives, _          = _get_step_embeddings(inputs_true,      step_ids_true)
        negatives, neg_orders = _get_step_embeddings(inputs_corrupted,  step_ids_corrupt)

        for H in positives:
            cos = F.cosine_similarity(H[:-1], H[1:], dim=-1)   # [S-1]
            total_loss  = total_loss + (1.0 - cos).mean()
            num_samples += 1

        for H, order in zip(negatives, neg_orders):
            order_tensor        = torch.tensor(order, device=H.device)
            mask_invalid        = order_tensor[:-1] > order_tensor[1:]
            if mask_invalid.any():
                cos = F.cosine_similarity(H[:-1][mask_invalid], H[1:][mask_invalid], dim=-1)
                total_loss  = total_loss + torch.relu(cos - self.alpha).mean()
                num_samples += 1

        return total_loss / (num_samples + 1e-9)


# ---------------------------------------------------------------------------
# 6.  KL Divergence Loss  — unchanged
# ---------------------------------------------------------------------------

class KLDivergenceLoss(nn.Module):
    def __init__(self, ref_model):
        super().__init__()
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def forward(self, student_logits, input_ids, attention_mask):
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()

        log_probs_student = F.log_softmax(shift_student_logits, dim=-1)
        probs_ref = F.softmax(shift_ref_logits, dim=-1)

        kl_per_token = F.kl_div(log_probs_student, probs_ref, reduction='none').sum(dim=-1)

        num_valid_tokens = shift_mask.sum()
        if num_valid_tokens > 0:
            return (kl_per_token * shift_mask).sum() / num_valid_tokens
        return torch.tensor(0.0, device=input_ids.device)
