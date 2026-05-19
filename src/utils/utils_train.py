# src/utils_train.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from peft import PeftModel
from utils.loss_functions import grad_reverse
import sys


def compute_forward_bundle(
    args,
    model,
    batch: dict[str, torch.Tensor],
):
    """
    Central forward dispatch.

    Returns:
        logits:   [B,T,V] or None
        lhs:      [B,T,D] or None   (standard forward, with wpe)
        lhs_mml:  [B,T,D] or None   (what you feed to MML; may be no-pos)
    """
    logits = None
    lhs = None

    # ----- Standard paths (CLM / Pooled-CLM / KL / MML / pos-adv / STP) ----
    # STP now uses real vocab tokens — no special forward pass needed.
    need_standard = (
        args.use_clm
        or args.use_stp
        or args.use_kl
        or (args.save_heatmaps and not args.no_pos_mml)
        or (args.use_mml and not args.no_pos_mml)
        or args.use_cos
        or args.use_grl
    )
    # TODO: make this part more flexible and move from this function to a dedicated model lhs/logits handler
    if need_standard:
        from utils.utils_model import SmolLM2WithAbsPE
        if isinstance(model, SmolLM2WithAbsPE):
            # Go through wrapper forward to inject absolute PE
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attn_mask'],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            lhs = outputs.hidden_states[-1]
            logits = outputs.logits
        else:
            # Unwrap PeftModel (LoRA) to reach the underlying CausalLM,
            # then split backbone/head so we avoid output_hidden_states=True.
            # LoRA layers are injected in-place, so the backbone still uses them.
            
            causal_lm = model.base_model.model if isinstance(model, PeftModel) else model

            backbone = causal_lm.base_model
            lm_head = causal_lm.get_output_embeddings()

            outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch['attn_mask'],
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            lhs = outputs.last_hidden_state
            logits = lm_head(lhs)
    return logits, lhs


def log_probe_stats(args, step: int, mml: float, pos_ce: float, pos_acc):
    """
    Explicit stdout flush for SLURM.
    """
    if not args.use_grl:
        return
    if pos_acc is None:
        return
    if step % args.log_interval != 0:
        return

    msg = (
        f"[step {step}] "
        f"MML={mml:.4f} "
        f"POS_CE={pos_ce:.4f} "
        f"POS_ACC={float(pos_acc):.4f}"
    )
    print(msg, file=sys.stdout, flush=True)





def pool_steps_and_posbins(lhs, step_ids, attn_mask, n_bins: int):
    B, T, D = lhs.shape
    H_list, y_list = [], []

    for b in range(B):
        sids = step_ids[b]
        step_order = sids[sids != 0]
        seen = set()
        sequential_order = [x.item() for x in step_order if not (x.item() in seen or seen.add(x.item()))]
        if len(sequential_order) < 1:
            continue

        T_eff = int(attn_mask[b].sum().item())

        for s in sequential_order:
            mask = (sids == s)
            idx = torch.where(mask)[0]
            if idx.numel() == 0:
                continue
            h = lhs[b][mask].mean(dim=0)
            H_list.append(h)
            first_pos = idx.min().item()
            pos_bin = int(first_pos * n_bins / max(T_eff, 1))
            pos_bin = min(pos_bin, n_bins - 1)
            y_list.append(pos_bin)

    if not H_list:
        return None, None

    H = torch.stack(H_list, dim=0)
    y = torch.tensor(y_list, device=lhs.device, dtype=torch.long)
    return H, y


def compute_pos_adv_loss(args, pos_head, lhs, batch):
    device = lhs.device
    pos_loss = torch.tensor(0.0, device=device)
    pos_acc = None

    if not args.use_grl:
        return pos_loss, pos_acc

    H_steps, y_pos = pool_steps_and_posbins(
        lhs, batch["step_indices"], batch["attn_mask"], args.pos_bins,
    )

    if H_steps is None:
        return pos_loss, pos_acc

    feats = grad_reverse(H_steps, args.grl_lambda)
    pos_logits = pos_head(feats)
    pos_loss = F.cross_entropy(pos_logits, y_pos)

    with torch.no_grad():
        pos_acc = (pos_logits.argmax(dim=-1) == y_pos).float().mean()

    return pos_loss, pos_acc

