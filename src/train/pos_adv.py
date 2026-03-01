# src/train/pos_adv.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from loss_functions import grad_reverse


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

    if not args.use_pos_adv:
        return pos_loss, pos_acc

    H_steps, y_pos = pool_steps_and_posbins(
        lhs, batch["step_indices_mml"], batch["attn_mask"], args.pos_bins,
    )

    if H_steps is None:
        return pos_loss, pos_acc

    feats = grad_reverse(H_steps, args.grl_lambda)
    pos_logits = pos_head(feats)
    pos_loss = F.cross_entropy(pos_logits, y_pos)

    with torch.no_grad():
        pos_acc = (pos_logits.argmax(dim=-1) == y_pos).float().mean()

    return pos_loss, pos_acc
