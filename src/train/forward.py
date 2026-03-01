# src/train/forward.py
from __future__ import annotations
import torch
from utils_model import forward_no_pos_gpt2


@torch.no_grad()
def _ensure_hidden_states(outputs) -> None:
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError(
            "Model outputs do not include hidden_states. "
            "Make sure output_hidden_states=True."
        )


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
    lhs_mml = None

    # ----- Standard paths (CLM / Pooled-CLM / KL / MML / pos-adv / STP) ----
    # STP now uses real vocab tokens — no special forward pass needed.
    need_standard = (
        args.use_clm
        or args.use_stp
        or args.use_kl
        or (args.save_heatmaps and not args.no_pos_mml)
        or (args.use_mml and not args.no_pos_mml)
        or args.use_pos_adv
    )

    if need_standard:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
        )
        logits = outputs.logits
        _ensure_hidden_states(outputs)
        lhs = outputs.hidden_states[-1]

    if args.use_mml:
        if args.no_pos_mml:
            out_np = forward_no_pos_gpt2(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attn_mask"],
                output_hidden_states=True,
            )
            lhs_mml = out_np.last_hidden_state
        else:
            lhs_mml = lhs

    return logits, lhs, lhs_mml
