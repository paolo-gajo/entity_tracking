# src/train/logging.py
from __future__ import annotations
import sys

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
