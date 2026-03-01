# cat_bench_regression_step_tokens.py
#
# Linear probe on CaT-Bench using step token hidden states.
# For each (step_a, step_b) pair, extracts the hidden state at the
# <step_a> and <step_b> token positions, builds feature vector
# [emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], trains logistic regression.

import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import os
import json
import argparse
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
    Build input_ids with <step_j> appended after each step's content.
    Returns input_ids, step_token_positions dict, attention_mask.
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
# Feature extraction
# -------------------------

@torch.no_grad()
def extract_features(df, tokenizer, model, step_token_ids, device, max_len):
    """
    For each sample, build input with step tokens, extract hidden states
    at the two queried step token positions, and build feature vector.
    """
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        steps = row['steps']
        n_steps = len(steps)
        if n_steps < 2:
            continue
        if n_steps > len(step_token_ids):
            continue

        idx_a, idx_b = row['step_pair_idx_asked_about']
        if not (0 <= idx_a < n_steps and 0 <= idx_b < n_steps):
            continue

        input_ids, stp_positions, attn_mask = build_step_token_input(
            steps, tokenizer, step_token_ids, device
        )

        if input_ids.numel() == 0:
            continue
        if input_ids.numel() > max_len:
            continue

        out = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0))
        lhs = out.last_hidden_state.squeeze(0)  # [T, D]

        emb_a = lhs[stp_positions[idx_a]]
        emb_b = lhs[stp_positions[idx_b]]

        # Feature: [A, B, A-B, A*B]
        feat = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=0)
        features.append(feat.cpu().numpy())
        labels.append(int(row['label']))

    if not features:
        return np.array([]), np.array([])

    return np.array(features), np.array(labels)

# -------------------------
# Saving
# -------------------------

def get_model_info(model_path, args, task_name="cat_bench_regression_step_tokens"):
    train_conf_path = os.path.join(model_path, "train_config.json")
    sample_dir = f"samples={args.sample_type}"

    if os.path.exists(train_conf_path):
        with open(train_conf_path, "r", encoding="utf8") as f:
            train_config_raw = json.load(f)
        train_config = setup_config(train_config_raw)
        model_save_dir = os.path.normpath(train_config["model_save_dir"])
        num_steps = str(train_config.get("num_steps", 0))
        rel = os.path.relpath(model_save_dir, start=os.path.normpath("./models"))
        save_path = os.path.join("./results", task_name, sample_dir, rel, num_steps)
        return save_path, train_config

    train_config = {"num_steps": 0}
    model_leaf = os.path.basename(os.path.normpath(model_path))
    save_path = os.path.join("./results", task_name, sample_dir, "baseline", model_leaf, "0")
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

def main(args):
    data_path_train = './data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json'
    data_path_test = './data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Load data once
    with open(data_path_train, 'r') as f:
        df_train = pd.DataFrame(json.load(f))
    with open(data_path_test, 'r') as f:
        df_test = pd.DataFrame(json.load(f))

    # Filter by sample type
    if args.sample_type != 'all':
        df_train = df_train[df_train['type'] == args.sample_type].reset_index(drop=True)
        df_test = df_test[df_test['type'] == args.sample_type].reset_index(drop=True)

    for m in model_list:
        model_name = m["path"]
        save_path, train_config = get_model_info(model_name, args)
        result_file = os.path.join(save_path, "results.json")
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        model, tokenizer, step_token_ids = setup_model(model_name, device, args.stp_max_steps)
        print(f"Using {len(step_token_ids)} step tokens")

        max_len = args.max_len
        if "gpt2" in model_name.lower():
            max_len = min(max_len, getattr(model.config, "n_positions", 1024))

        # Extract features
        print("--- Extracting train features ---")
        X_train, y_train = extract_features(df_train, tokenizer, model, step_token_ids, device, max_len)
        print("--- Extracting test features ---")
        X_test, y_test = extract_features(df_test, tokenizer, model, step_token_ids, device, max_len)

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        if X_train.size == 0 or X_test.size == 0:
            print("No features extracted, skipping")
            continue

        # Train logistic regression probe
        print("Training logistic regression probe...")
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        results = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "acc": float(np.mean(preds == y_test)),
            "f1_macro": float(f1_score(y_test, preds, average='macro')),
            "f1_binary": float(f1_score(y_test, preds, average='binary')),
        }
        if len(np.unique(y_test)) == 2:
            results["roc_auc"] = float(roc_auc_score(y_test, probs))

        report = classification_report(y_test, preds, digits=4)

        if args.verbose_results:
            print(f"Model: {model_name}")
            print(report)
            print(f"Macro F1: {results['f1_macro']:.4f}")
            if "roc_auc" in results:
                print(f"ROC AUC: {results['roc_auc']:.4f}")

        results["report"] = report

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--stp_max_steps", type=int, default=15)
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--sample_type", default="real", choices=["real", "all"], help="'real' for only real samples, 'all' for real+switched")

    args = parser.parse_args()
    main(args)
