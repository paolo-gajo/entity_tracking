# src/train/forward.py
from __future__ import annotations
import torch

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
    if need_standard:
        # TODO: make this part more flexible and move from this function to a dedicated model lhs/logits handler
        if ('gpt2' in args.model_name.lower() or
            'neo' in args.model_name.lower()):
            backbone = model.transformer
            lm_head = model.lm_head
            
            outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch['attn_mask'],
                output_hidden_states=False, 
                use_cache=False,
                return_dict=True
            )
            lhs = outputs.last_hidden_state
            logits = lm_head(lhs)

        elif (
            'qwen' in args.model_name.lower() or
            'opt' in args.model_name.lower() or
            'smol' in args.model_name.lower()
            ):
            from utils_model import SmolLM2WithAbsPE
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
                backbone = model.model
                lm_head = model.lm_head

                outputs = backbone(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attn_mask'],
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True
                )
                lhs = outputs.last_hidden_state
                logits = lm_head(lhs)
    return logits, lhs