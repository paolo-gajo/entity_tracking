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
                  lhs_mml,
                  pos_loss=None,
                  stp_loss=None,
                  ):
    if args.use_clm:
        assert logits is not None
    if args.use_kl:
        assert logits is not None

    if args.use_clm:
        if args.pool_clm:
            # Pooled CLM requires step indices from the completion side
            causal_lm_loss = causal_lm_loss_fn(
                logits, batch['input_ids'], batch['loss_mask'], batch['step_indices_mml']
            )
        else:
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
    stp_loss = stp_loss if stp_loss is not None else torch.tensor(0.0, device=device)

    total_loss = (
        causal_lm_loss  * args.clm_lambda +
        kl_loss         * args.kl_lambda +
        max_margin_loss * args.mml_lambda +
        pos_loss        * args.pos_lambda +
        stp_loss        * args.stp_lambda
    )

    return {
        'causal_lm_loss':  causal_lm_loss,
        'kl_loss':         kl_loss,
        'max_margin_loss': max_margin_loss,
        'pos_loss':        pos_loss,
        'stp_loss':        stp_loss,
        'total_loss':      total_loss,
    }

class CausalLMLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, input_ids, loss_mask):
        labels = input_ids.clone()
        labels[loss_mask == 0] = self.ignore_index

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


# ---------------------------------------------------------------------------
# 2.  Pooled Causal LM Loss
#
#     L_clm = (1/B) sum_b sum_{j=1}^{N} [ -(1/|S_j|) sum_{t in S_j} log p(s_t | s_{<t}) ]
#
#     Instead of weighting every token uniformly, we first average per-token CE
#     *within* each step, then sum over steps.  This prevents long steps from
#     dominating the gradient and forces the model to allocate capacity to
#     the first few tokens of each step (the "classification" tokens) that
#     induction heads could otherwise shortcut.
# ---------------------------------------------------------------------------

class PooledCausalLMLoss(nn.Module):
    """
    Per-step pooled cross-entropy.

    For each step j in each sample, the loss contribution is the *mean* CE
    across that step's tokens, and the sample loss is the *sum* over steps.
    We then average over samples in the batch.

    Signature is intentionally a superset of CausalLMLoss so that the caller
    in gather_losses can branch on args.pool_clm.
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, input_ids, loss_mask, step_indices):
        """
        Args:
            logits:       [B, T, V]
            input_ids:    [B, T]
            loss_mask:    [B, T]   1 = completion token
            step_indices: [B, T]   step id per token (1..N in the completion,
                                   0 elsewhere including prefix and padding)
        Returns:
            Scalar loss.
        """
        B, T, V = logits.shape

        # --- next-token shift --------------------------------------------------
        shift_logits = logits[:, :-1, :].contiguous()           # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()            # [B, T-1]
        shift_mask   = loss_mask[:, 1:].contiguous().float()    # [B, T-1]
        shift_steps  = step_indices[:, 1:].contiguous()         # [B, T-1]

        # per-token CE, no reduction
        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction='none',
        ).view(B, T - 1)                                       # [B, T-1]

        per_token_ce = per_token_ce * shift_mask                # mask padding / prefix

        # --- pool by step, sum over steps, average over batch ------------------
        batch_loss = torch.tensor(0.0, device=logits.device)
        n_samples = 0

        for b in range(B):
            active = shift_mask[b].bool()
            if not active.any():
                continue

            active_steps = shift_steps[b][active]
            unique_steps = active_steps.unique()
            unique_steps = unique_steps[unique_steps > 0]

            if unique_steps.numel() == 0:
                # Fallback: no step annotation on the completion side.
                # This can happen with prompt types that don't assign step indices
                # to completion tokens.  Fall back to flat mean over masked tokens.
                batch_loss = batch_loss + per_token_ce[b][active].mean()
                n_samples += 1
                continue

            sample_loss = torch.tensor(0.0, device=logits.device)
            for s in unique_steps:
                step_mask = (shift_steps[b] == s) & active
                sample_loss = sample_loss + per_token_ce[b][step_mask].mean()

            batch_loss = batch_loss + sample_loss
            n_samples += 1

        if n_samples == 0:
            return torch.tensor(0.0, device=logits.device)
        return batch_loss / n_samples


# ---------------------------------------------------------------------------
# 3.  Step Token Prediction Loss  (Section 3.2 of the paper)
#
#     At each completion position, predict which learned step-token embedding
#     comes next via classification over M pre-instantiated embeddings.
# ---------------------------------------------------------------------------

class StepTokenLoss(nn.Module):
    """
    Cross-entropy over M step-token classes.

    Operates on the raw hidden states and a lightweight classification head
    (StepTokenHead).  Positions where stp_labels == -100 are ignored.
    """

    def __init__(self, step_token_head):
        super().__init__()
        self.head = step_token_head

    def forward(self, hidden_states, stp_labels):
        """
        Args:
            hidden_states: [B, T, D]  — last hidden states from the transformer
            stp_labels:    [B, T]     — target step-token class (0 … M-1)
                                        at each position;  -100 = ignore.
        Returns:
            Scalar CE loss (averaged over valid positions).
        """
        logits = self.head(hidden_states)                       # [B, T, M]
        B, T, M = logits.shape

        # Clamp step labels to valid range [0, M-1] (ignoring -100 which we restore)
        valid_mask = stp_labels != -100
        clamped_labels = stp_labels.clone()
        clamped_labels[valid_mask] = clamped_labels[valid_mask].clamp(min=0, max=M - 1)

        return F.cross_entropy(
            logits.view(-1, M),
            clamped_labels.view(-1),
            ignore_index=-100,
        )


# ---------------------------------------------------------------------------
# 4.  Max-Margin Loss  — unchanged from your original
# ---------------------------------------------------------------------------

class MaxMarginLoss(nn.Module):
    """
    Computes the max-margin order embedding loss for directed acyclic graph (DAG) topologies.
    """
    def __init__(self, alpha, activations):
        super().__init__()
        self.alpha = alpha
        self.activations = activations

    @staticmethod
    def E(x, y):
        return torch.relu(y - x).pow(2).mean(dim=-1)

    def forward(self, inputs, step_ids, binary_labels):
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
        H_list = []
        order_list = []
        for b in range(hidden_states.size(0)):
            step_order_tensor = step_ids[b]
            step_order = step_order_tensor[step_order_tensor != 0]

            seen = set()
            sequential_order = [x.item() for x in step_order if not (x.item() in seen or seen.add(x.item()))]

            if len(sequential_order) < 2:
                continue

            h_list = []
            for s in sequential_order:
                mask = (step_ids[b] == s)
                h_list.append(hidden_states[b][mask].mean(dim=0))

            H = torch.stack(h_list)
            H_list.append(H)
            order_list.append(sequential_order)

        return H_list, order_list


# ---------------------------------------------------------------------------
# 5.  KL Divergence Loss  — unchanged
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
