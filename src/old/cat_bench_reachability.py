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
import os
from utils_sys import setup_config

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
    # E(x, y) = 0 \iff x \prec y
    # S[i,j] = -||ReLU(H[i] - H[j])||^2
    # the embedding elements of the latter h \in H
    # should be bigger than the earlier ones
    # so S = 0 should generally be more common whenever j > i
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)  # [N,N,D]
    penalty = torch.relu(diff).pow(2).sum(dim=-1)       # [N,N]
    return -penalty

def widest_path_closure(S: torch.Tensor) -> torch.Tensor:
    """Widest-path closure: R[i,j] = max over all paths i->...->j of
    min edge weight along the path. Used as a soft reachability score."""
    R = S.clone()
    n = R.shape[0]
    R.fill_diagonal_(-1e9)
    for k in range(n):
        via = torch.minimum(R[:, k].unsqueeze(1), R[k, :].unsqueeze(0))
        R = torch.maximum(R, via)
    R.fill_diagonal_(-1e9)
    return R

def get_model_info(model_path, args, task_name="cat_bench_reachability_step_tokens"):
    train_conf_path = os.path.join(model_path, "train_config.json")

    if os.path.exists(train_conf_path):
        with open(train_conf_path, "r", encoding="utf8") as f:
            train_config_raw = json.load(f)
        train_config = setup_config(train_config_raw)
        model_save_dir = os.path.normpath(train_config["model_save_dir"])
        num_steps = str(train_config.get("num_steps", 0))
        rel = os.path.relpath(model_save_dir, start=os.path.normpath("./models"))
        save_path = os.path.join("./results", task_name, rel, f"samples={args.sample_type}", num_steps)
        return save_path, train_config

    train_config = {"num_steps": 0}
    model_leaf = os.path.basename(os.path.normpath(model_path))
    save_path = os.path.join("./results", task_name, "baseline", model_leaf, f"activations={args.activations}", f"samples={args.sample_type}", "0")
    return save_path, train_config

def save_results_to_disk(results, save_path, train_config, args):
    os.makedirs(save_path, exist_ok=True)
    out_dict = {
        "train_config": train_config,
        "eval_config": vars(args),
        "results": results,
    }
    json_path = os.path.join(save_path, "results.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {json_path}")

@torch.no_grad()
def eval_catbench(model, tokenizer, df, device, activations, max_len):
    y_true, y_score = [], []

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
        R = widest_path_closure(S)

        a, b = idx_a, idx_b
        if direction == "after":
            score = float(R[a, b].item())
        elif direction == "before":
            score = float(R[b, a].item())
        else:
            continue

        y_true.append(label)
        y_score.append(score)

    return np.array(y_true), np.array(y_score)

def setup_model(model_name, device):
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Walk model_dir for checkpoints or treat as single model
    if not os.path.exists(args.model_dir):
        model_list = [{"path": args.model_dir, "num_steps": 0}]
    else:
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for F in files:
                if F == "train_config.json":
                    with open(os.path.join(root, F), "r", encoding="utf8") as f:
                        num_steps = json.load(f)["num_steps"]
                    model_list.append({"path": root, "num_steps": num_steps})
        if not model_list:
            model_list = [{"path": args.model_dir, "num_steps": 0}]
        model_list = sorted(model_list, key=lambda x: x["num_steps"])

    # Filter by sample type
    if args.sample_type != 'all':
        df = df[df['type'] == args.sample_type].reset_index(drop=True)

    for m in model_list:
        model_name = m["path"]
        save_path, train_config = get_model_info(model_name, args, task_name="cat_bench_reachability_pooled")
        result_file = os.path.join(save_path, "results.json")
        
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        model, tokenizer = setup_model(model_name, device)

        if "gpt2" in model_name.lower():
            args.max_len = min(args.max_len, getattr(model.config, "n_positions", 1024))

        y_true, y_score = eval_catbench(
            model=model,
            tokenizer=tokenizer,
            df=df,
            device=device,
            activations=args.activations,
            max_len=args.max_len,
        )

        # Binary predictions: threshold at median score
        threshold = float(np.median(y_score))
        y_pred = (y_score >= threshold).astype(int)

        results = {"n": int(len(y_true))}

        if len(np.unique(y_true)) == 2:
            signed_roc_auc = float(roc_auc_score(y_true, y_score))
            unsigned_roc_auc = max(signed_roc_auc, 1.0 - signed_roc_auc)
            
            # Determine signal polarity
            is_inverted = signed_roc_auc < 0.5
            
            # Conditional thresholding
            threshold = float(np.median(y_score))
            if is_inverted:
                y_pred = (y_score <= threshold).astype(int)
            else:
                y_pred = (y_score >= threshold).astype(int)

            results["signed_roc_auc"] = signed_roc_auc
            results["unsigned_roc_auc"] = unsigned_roc_auc
            results["signed_avg_precision"] = float(average_precision_score(y_true, y_score))
            
            # Invert scores for unsigned AP if polarity is inverted
            y_score_unsigned = -y_score if is_inverted else y_score
            results["unsigned_avg_precision"] = float(average_precision_score(y_true, y_score_unsigned))

        else:
            # Fallback for mono-class edge cases
            threshold = float(np.median(y_score))
            y_pred = (y_score >= threshold).astype(int)

        results["acc"] = float(accuracy_score(y_true, y_pred))
        results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if args.verbose_results:
            print(f"n = {results['n']}")
            print(f"acc = {results['acc']:.4f}")
            print(f"f1  = {results['f1']:.4f}")
            if "signed_roc_auc" in results:
                print(f"signed_roc   = {results['signed_roc_auc']:.4f}")
                print(f"unsigned_roc = {results['unsigned_roc_auc']:.4f}")
                print(f"signed_pr    = {results['signed_avg_precision']:.4f}")
                print(f"unsigned_pr  = {results['unsigned_avg_precision']:.4f}")
            print(classification_report(y_true, y_pred, digits=4))

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--data_path", default="./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json")
    parser.add_argument("--activations", default="real", choices=["real", "non-negative"])
    parser.add_argument("--max_len", type=int, default=1024)

    parser.add_argument("--sample_type", default="real", choices=["real", "all"], help="'real' for only real samples, 'all' for real+switched")
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=1, type=int)

    args = parser.parse_args()
    main(args)