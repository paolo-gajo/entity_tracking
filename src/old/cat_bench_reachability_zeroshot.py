# cat_bench_reachability_zeroshot.py
import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report
import json
import argparse

# -------------------------
# Encoding / pooling (sims-style)
# -------------------------

def build_concat_inputs_from_steps(steps, tokenizer, device):
    ids_list, idx_list = [], []
    for j, step in enumerate(steps, start=1):
        step_ids = tokenizer.encode(" " + step.strip(), add_special_tokens=False)
        ids_list.extend(step_ids)
        idx_list.extend([j] * len(step_ids))

    if len(ids_list) == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    step_indices = torch.tensor(idx_list, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, step_indices, attention_mask

def pool_steps(lhs, step_indices, n_steps):
    h_steps = []
    for j in range(1, n_steps + 1):
        pos = torch.where(step_indices == j)[0]
        if pos.numel() == 0:
            h = torch.zeros((lhs.shape[-1],), device=lhs.device, dtype=lhs.dtype)
        else:
            h = lhs[pos].mean(dim=0)
        h_steps.append(h)
    return torch.stack(h_steps, dim=0)  # [N,D]

def directed_score_matrix(H_steps):
    # S[i,j] = -||ReLU(H[i] - H[j])||^2  (higher is "better"/more compatible)
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)  # [N,N,D]
    penalty = torch.relu(diff).pow(2).sum(dim=-1)       # [N,N]
    return -penalty

# -------------------------
# Turn scores into predicted adjacency
# -------------------------

def scores_to_adj_topk(S, k):
    """
    For each i, add edges i->j for top-k highest S[i,j] (excluding i).
    """
    S = S.detach().cpu()
    N = S.shape[0]
    A = torch.zeros((N, N), dtype=torch.uint8)
    for i in range(N):
        row = S[i].clone()
        row[i] = -1e9  # exclude self
        # topk may exceed N-1 for short N
        kk = min(k, N - 1)
        if kk <= 0:
            continue
        idx = torch.topk(row, kk).indices
        A[i, idx] = 1
    return A.numpy().astype(np.uint8)

def scores_to_adj_tau(S, tau):
    """
    Use energy E = -S. Edge i->j if E[i,j] <= tau.
    """
    E = (-S).detach().cpu().numpy()
    A = (E <= tau).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A

def transitive_closure_adj(A):
    """
    A: [N,N] uint8 adjacency
    Returns reachability R: [N,N] uint8, no self loops.
    """
    N = A.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    src, tgt = np.where(A == 1)
    edges = [(int(i), int(j)) for i, j in zip(src, tgt) if i != j]
    G.add_edges_from(edges)

    TC = nx.transitive_closure(G)
    R = nx.to_numpy_array(TC, nodelist=list(range(N))).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

# -------------------------
# Evaluation
# -------------------------

@torch.no_grad()
def eval_catbench(model, tokenizer, df, device, activations, max_len, edge_mode,
                #   k,
                  tau):
    y_true, y_pred, y_score = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        steps = row["steps"]
        n_steps = len(steps)
        if n_steps < 2:
            continue

        idx_a, idx_b = row["step_pair_idx_asked_about"]
        direction = row.get("direction", "after")
        label = int(row["label"])

        if not (0 <= idx_a < n_steps and 0 <= idx_b < n_steps):
            continue

        input_ids, step_indices, attention_mask = build_concat_inputs_from_steps(steps, tokenizer, device)
        if input_ids.numel() == 0:
            continue

        # Exclude sequences exceeding the context window to prevent zero-vector injection
        if input_ids.numel() > max_len:
            continue

        out = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

        lhs = out.last_hidden_state.squeeze(0)  # [T,D]
        if activations == "non-negative":
            lhs = torch.abs(lhs)

        H = pool_steps(lhs, step_indices, n_steps)     # [N,D]
        S = directed_score_matrix(H)                   # [N,N]

        # Build predicted adjacency Â
        # if edge_mode == "topk":
        #     Ahat = scores_to_adj_topk(S, k=k)
        if edge_mode == "tau":
            Ahat = scores_to_adj_tau(S, tau=tau)
        else:
            raise ValueError(edge_mode)

        # Reachability (transitive closure)
        Rhat = transitive_closure_adj(Ahat)

        a, b = idx_a, idx_b

        # Prediction from reachability
        if direction == "after":
            pred = int(Rhat[a, b] == 1)
            score = float(S[a, b].item())     # continuous proxy (bigger => more likely)
        elif direction == "before":
            pred = int(Rhat[b, a] == 1)
            score = float(S[b, a].item())
        else:
            continue

        y_true.append(label)
        y_pred.append(pred)
        y_score.append(score)

    return np.array(y_true), np.array(y_pred), np.array(y_score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="openai-community/gpt2")
    ap.add_argument("--data_path", default="./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json")
    ap.add_argument("--activations", default="real", choices=["real", "non-negative"])
    ap.add_argument("--max_len", type=int, default=1024)

    # How to create adjacency from S
    ap.add_argument("--edge_mode", default="tau", choices=[
        # "topk",
        "tau",
        ])
    # ap.add_argument("--k", type=int, default=2, help="top-k outgoing edges per node (edge_mode=topk)")
    ap.add_argument("--tau", type=float, default=5.0, help="energy threshold (edge_mode=tau): edge if -S <= tau")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(args.model_dir).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPT2 length cap
    if "gpt2" in args.model_dir:
        args.max_len = min(args.max_len, getattr(model.config, "n_positions", 1024))

    with open(args.data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["type"] == "real"].reset_index(drop=True)

    y_true, y_pred, y_score = eval_catbench(
        model=model,
        tokenizer=tokenizer,
        df=df,
        device=device,
        activations=args.activations,
        max_len=args.max_len,
        edge_mode=args.edge_mode,
        # k=args.k,
        tau=args.tau,
    )

    print("n =", len(y_true))
    print("acc =", accuracy_score(y_true, y_pred))
    print("f1  =", f1_score(y_true, y_pred))

    if len(np.unique(y_true)) == 2:
        print("roc =", roc_auc_score(y_true, y_score))
        print("pr  =", average_precision_score(y_true, y_score))

    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()