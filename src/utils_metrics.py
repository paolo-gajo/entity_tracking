import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def get_auc(S: torch.Tensor, A: np.ndarray, verbose = False) -> float:
    """
    Compute ROC–AUC between continuous scores S and binary adjacency A.
    Diagonal is excluded. S is torch (n,n). A is numpy (n,n).
    """
    S_np = S.detach().cpu().numpy()
    A_np = A.astype(int)

    n = A_np.shape[0]
    assert S_np.shape == (n, n), f"Shape mismatch: S {S_np.shape} vs A {(n,n)}"

    mask = ~np.eye(n, dtype=bool)
    y_true = A_np[mask]
    y_score = S_np[mask]

    # AUC undefined if only one class present
    if np.unique(y_true).size < 2:
        return float("nan")
    score = roc_auc_score(y_true, y_score)
    if verbose:
        print(A_np)
        print(S_np)
        print(score)
    return score