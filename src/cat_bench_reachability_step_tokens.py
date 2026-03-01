# cat_bench_reachability_step_tokens.py
#
# CaT-Bench evaluation using step token hidden states.
# Same philosophy as sims_step_tokens.py: append <step_j> after each step's
# content, extract hidden state at that position, compute Vendrov directed score.
# Evaluates individual (a,b) pairs from CaT-Bench questions.

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report
import json
import argparse
import os
from utils_sys import setup_config

# -------------------------
# Model setup (same as sims_step_tokens.py)
# -------------------------

def get_step_token_ids(tokenizer, max_steps):
    ids = []
    for i in range(max_steps):
        tok = f"<step_{i}>"
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            break
        ids.append(tid)
    return ids

def setup_model(model_name, device, stp_max_steps=15):
    print(f'Loading model: {model_name}')
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if 'gpt2' in model_name.lower():
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id

    # Add step tokens if not present (random-init baseline)
    step_token_ids = get_step_token_ids(tokenizer, stp_max_steps)
    if len(step_token_ids) == 0:
        step_tokens = [f"<step_{i}>" for i in range(stp_max_steps)]
        tokenizer.add_tokens(step_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        step_token_ids = get_step_token_ids(tokenizer, stp_max_steps)
        print(f"Added {len(step_token_ids)} step tokens to baseline model (random init)")

    model = model.to(device)
    model.eval()
    return model, tokenizer, step_token_ids

# -------------------------
# Input construction with step tokens
# -------------------------

def build_step_token_input(steps, tokenizer, step_token_ids, device):
    """
    Given a list of step strings, build token-level input_ids with <step_j>
    appended after each step's content (j = input order).

    Returns:
        input_ids:            [T] token ids
        step_token_positions: dict  step_index (0-based) -> position in input_ids
        attention_mask:       [T] all ones
    """
    input_ids = []
    step_token_positions = {}

    for j, step_text in enumerate(steps):
        step_tok_ids = tokenizer.encode(" " + step_text.strip(), add_special_tokens=False)
        input_ids.extend(step_tok_ids)

        if j < len(step_token_ids):
            stp_id = step_token_ids[j]
        else:
            stp_id = step_token_ids[-1]

        step_token_positions[j] = len(input_ids)
        input_ids.append(stp_id)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, step_token_positions, attention_mask

# -------------------------
# Scoring
# -------------------------

def directed_score_matrix(H_steps):
    """S[i,j] = -||ReLU(H[i] - H[j])||^2"""
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)
    penalty = torch.relu(diff).pow(2).sum(dim=-1)
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

# -------------------------
# Evaluation
# -------------------------

@torch.no_grad()
def eval_catbench(model, tokenizer, step_token_ids, df, device, activations, max_len):
    y_true, y_score = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        steps = row["steps"]
        n_steps = len(steps)
        if n_steps < 2:
            continue

        # Skip if more steps than step tokens
        if n_steps > len(step_token_ids):
            continue

        idx_a, idx_b = row["step_pair_idx_asked_about"]
        direction = row.get("direction", "after")
        label = int(row["label"])

        if not (0 <= idx_a < n_steps and 0 <= idx_b < n_steps):
            continue

        # Build input with step tokens
        input_ids, stp_positions, attn_mask = build_step_token_input(
            steps, tokenizer, step_token_ids, device
        )

        if input_ids.numel() == 0:
            continue
        if input_ids.numel() > max_len:
            continue

        out = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0))
        lhs = out.last_hidden_state.squeeze(0)  # [T, D]

        if activations == "non-negative":
            lhs = torch.abs(lhs)

        # Extract step token hidden states
        H_list = []
        for j in range(n_steps):
            pos = stp_positions[j]
            H_list.append(lhs[pos])
        H_steps = torch.stack(H_list)  # [N, D]

        # Score matrix + widest-path closure (soft reachability)
        S = directed_score_matrix(H_steps)
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

# -------------------------
# Saving
# -------------------------

def get_model_info(model_path, args, task_name="cat_bench_step_tokens"):
    train_conf_path = os.path.join(model_path, "train_config.json")

    if os.path.exists(train_conf_path):
        with open(train_conf_path, "r", encoding="utf8") as f:
            train_config_raw = json.load(f)
        train_config = setup_config(train_config_raw)
        model_save_dir = os.path.normpath(train_config["model_save_dir"])
        num_steps = str(train_config.get("num_steps", 0))
        rel = os.path.relpath(model_save_dir, start=os.path.normpath("./models"))
        save_path = os.path.join("./results", task_name, rel, num_steps)
        return save_path, train_config

    train_config = {"num_steps": 0}
    model_leaf = os.path.basename(os.path.normpath(model_path))
    save_path = os.path.join("./results", task_name, "baseline", model_leaf, f"activations={args.activations}", "0")
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

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="openai-community/gpt2")
    ap.add_argument("--data_path", default="./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json")
    ap.add_argument("--activations", default="real", choices=["real", "non-negative"])
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stp_max_steps", type=int, default=15)
    ap.add_argument("--save_results", default=1, type=int)
    ap.add_argument("--verbose_results", default=1, type=int)
    ap.add_argument("--repeat", default=1, type=int)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["type"] == "real"].reset_index(drop=True)

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

    for m in model_list:
        model_name = m["path"]
        save_path, train_config = get_model_info(model_name, args)
        result_file = os.path.join(save_path, "results.json")
        
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        model, tokenizer, step_token_ids = setup_model(model_name, device, args.stp_max_steps)
        print(f"Using {len(step_token_ids)} step tokens")

        if "gpt2" in model_name.lower():
            args.max_len = min(args.max_len, getattr(model.config, "n_positions", 1024))

        y_true, y_score = eval_catbench(
            model=model,
            tokenizer=tokenizer,
            step_token_ids=step_token_ids,
            df=df,
            device=device,
            activations=args.activations,
            max_len=args.max_len,
        )

        # Binary predictions: threshold at median score
        threshold = float(np.median(y_score))
        y_pred = (y_score >= threshold).astype(int)

        results = {
            "n": int(len(y_true)),
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) == 2:
            results["roc_auc"] = float(roc_auc_score(y_true, y_score))
            results["avg_precision"] = float(average_precision_score(y_true, y_score))

        if args.verbose_results:
            print(f"n = {results['n']}")
            print(f"acc = {results['acc']:.4f}")
            print(f"f1  = {results['f1']:.4f}")
            if "roc_auc" in results:
                print(f"roc = {results['roc_auc']:.4f}")
                print(f"pr  = {results['avg_precision']:.4f}")
            print(classification_report(y_true, y_pred, digits=4))

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    main()
