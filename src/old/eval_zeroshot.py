# eval_zeroshot.py
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
import os
import json
import argparse
from sims import get_model_info

def build_concat_inputs_from_steps(steps, tokenizer, device):
    """
    Mimic sims.py: concatenate all step token ids into one sequence and
    produce a step_indices vector with the same length.
    Step indices are 1..N (0 is reserved).
    """
    ids_list = []
    idx_list = []
    cur = 0

    for j, step in enumerate(steps, start=1):
        step_text = " " + step.strip()   # keep your training delimiter behavior
        step_ids = tokenizer.encode(step_text, add_special_tokens=False)

        ids_list.extend(step_ids)
        idx_list.extend([j] * len(step_ids))
        cur += len(step_ids)

    if len(ids_list) == 0:
        # empty sample; return empty tensors
        return (
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    step_indices = torch.tensor(idx_list, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, step_indices, attention_mask


def pool_steps_last_hidden_state(lhs, step_indices, n_steps):
    """
    Pool per-step embeddings exactly like sims.get_step_embeddings:
    mean over token hidden states belonging to each step id.
    """
    h_steps = []
    for j in range(1, n_steps + 1):
        pos = torch.where(step_indices == j)[0]
        if pos.numel() == 0:
            # should not happen unless tokenization produced empty steps
            h = torch.zeros((lhs.shape[-1],), device=lhs.device, dtype=lhs.dtype)
        else:
            h = lhs[pos].mean(dim=0)
        h_steps.append(h)
    return torch.stack(h_steps, dim=0)  # [N, D]


def directed_score_matrix(H_steps):
    """
    Same as sims.compute_scores directed part:
    S[i,j] = -||ReLU(H[i] - H[j])||^2
    """
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)  # [N, N, D]
    penalty = torch.relu(diff).pow(2).sum(dim=-1)       # [N, N]
    return -penalty


@torch.no_grad()
def get_geometric_scores(batch_df, tokenizer, model, device, activations="real"):
    """
    For each row: build full S_directed using sims.py-style encoding,
    then select the entry for the asked pair (with direction handling).
    """
    scores = []

    for _, row in batch_df.reset_index(drop=True).iterrows():
        steps = row["steps"]
        idx_a, idx_b = row["step_pair_idx_asked_about"]
        direction = row.get("direction", "after")

        # Build concatenated sequence + step indices
        input_ids, step_indices, attention_mask = build_concat_inputs_from_steps(
            steps, tokenizer, device
        )

        # Guard
        n_steps = len(steps)
        if n_steps == 0:
            scores.append(np.nan)
            continue
        if not (0 <= idx_a < n_steps and 0 <= idx_b < n_steps):
            scores.append(np.nan)
            continue

        # Cap to model max length to match your previous code behavior
        max_len = getattr(model.config, "n_positions", 1024)
        if input_ids.numel() > max_len:
            input_ids = input_ids[:max_len]
            step_indices = step_indices[:max_len]
            attention_mask = attention_mask[:max_len]

        # Forward pass (AutoModel like sims.py)
        out = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        lhs = out.last_hidden_state.squeeze(0)  # [T, D]
        if activations == "non-negative":
            lhs = torch.abs(lhs)

        # Pool per step
        H_steps = pool_steps_last_hidden_state(lhs, step_indices, n_steps)  # [N, D]

        # Full directed matrix
        S = directed_score_matrix(H_steps)  # [N, N]

        # Map CaT indices (0-based) -> our pooled rows (also 0-based)
        a = idx_a
        b = idx_b

        # Semantics:
        # - "after": must b happen after a?  -> look at S[a,b]
        # - "before": must b happen before a? -> look at S[b,a]
        if direction == "after":
            score = S[a, b].item()
        elif direction == "before":
            score = S[b, a].item()
        else:
            scores.append(np.nan)
            continue

        scores.append(score)

    return np.array(scores, dtype=float)


def main(args):
    # Path to Test Data
    data_path = './data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists(args.model_dir):
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for F in files:
                if F == 'model.safetensors':
                    model_list.append(root)
    else:
        model_list = [args.model_dir]
    for model_name in model_list:
        print(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load JSON Data
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # print('direction', df['direction'].value_counts(dropna=False))
        bad = set(df['direction'].dropna().unique()) - {'after','before'}
        assert not bad, bad
        # print('question_type', df['question_type'].value_counts().head(20))
        # print(f"Types of sample: {df['type'].unique()}")
        # Optional: Filter for specific types if you want to inspect subsets
        # print(f'Only using {args.type} samples!')
        # df = df[df['type'] == args.type]    
        print(f"Evaluating {len(df)} samples...")

        all_scores = []
        
        batch_size = 16
        df['score'] = 0.0
        # df['objective'] = args.bigger_objective
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i : i+batch_size]
            scores = get_geometric_scores(batch,
                                        tokenizer,
                                        model,
                                        device,
                                        # bigger_objective = args.bigger_objective,
                                        activations = args.activations,
                                        )
            df.loc[i : i+batch_size-1, 'score'] = scores
            if len(scores) > 0:
                all_scores.append(scores)
                
        if not all_scores:
            print("No scores computed.")
            return
        
        df_real = df[df['type']=='real'].copy()
        # print(df_real.groupby('label')['score'].describe()[['mean','std','count']])
        for d in ['after','before']:
            sub = df_real[df_real['direction']==d]
            if sub['label'].nunique()==2:
                print(d, roc_auc_score(sub['label'], sub['score']), len(sub))

        # print("NaN scores:", df['score'].isna().mean(), df['score'].isna().sum())
        # print(df[df['score'].isna()].groupby(['type','direction','label']).size().head(20))

        all_scores_np = np.concatenate(all_scores, axis=0)
        
        # --- Compute Metrics ---
        # ROC AUC: Measures ranking quality
        all_labels = df['label'].values
        all_scores_np = df['score'].values
        roc_auc = roc_auc_score(all_labels, all_scores_np)
        # acc = accuracy_score(all_labels, all_scores_np)

        df_real = df[df['type'] == 'real']
        all_labels_real = df_real['label'].values
        all_scores_np_real = df_real['score'].values
        roc_auc_real = roc_auc_score(all_labels_real, all_scores_np_real)
        # acc_real = accuracy_score(all_labels_real, all_scores_np_real)

        df_switched = df[df['type'] == 'switched']
        all_labels_switched = df_switched['label'].values
        all_scores_np_switched = df_switched['score'].values
        roc_auc_switched = roc_auc_score(all_labels_switched, all_scores_np_switched)
        # acc_switched = accuracy_score(all_labels_switched, all_scores_np_switched)
        
        # PR AUC: Measures precision/recall trade-off (good for unbalanced)
        precision, recall, _ = precision_recall_curve(all_labels, all_scores_np)
        pr_auc = auc(recall, precision)

        precision_real, recall_real, _ = precision_recall_curve(all_labels_real, all_scores_np_real)
        pr_auc_real = auc(recall_real, precision_real)

        precision_switched, recall_switched, _ = precision_recall_curve(all_labels_switched, all_scores_np_switched)
        pr_auc_switched = auc(recall_switched, precision_switched)
        
        print("\n" + "="*40)
        print(f"ZERO-SHOT GEOMETRIC EVALUATION")
        print(f"Model: {model_name}")
        print("="*40)
        # print(f"ACC: {acc:.4f}")
        # print(f"ROC AUC: {roc_auc:.4f}")
        # print(f"PR AUC:  {pr_auc:.4f}")
        # print("="*40)
        # print(f"ACC REAL: {acc_real:.4f}")
        print(f"ROC AUC REAL: {roc_auc_real:.4f}")
        print(f"PR AUC REAL:  {pr_auc_real:.4f}")
        print("="*40)
        # print(f"ACC SWITCHED: {acc_switched:.4f}")
        # print(f"ROC AUC SWITCHED: {roc_auc_switched:.4f}")
        # print(f"PR AUC SWITCHED:  {pr_auc_switched:.4f}")
        # print("="*40)

        # Save Results
        save_path, train_config = get_model_info(model_name, args, task_name='cat_bench_zeroshot')
        os.makedirs(save_path, exist_ok=True)
        res_path = os.path.join(save_path, f"results.json")
        
        with open(res_path, 'w') as f:
            json.dump({
                'model': model_name,
                'num_steps': train_config['num_steps'],
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                # # 'acc': acc,
                'roc_auc_real': roc_auc_real,
                'pr_auc_real': pr_auc_real,
                # 'acc_real': acc_real,
                'roc_auc_switched': roc_auc_switched,
                'pr_auc_switched': pr_auc_switched,
                # 'acc_switched': acc_switched,
                'n_samples': len(df)
            }, f, indent=4)
            
        print(f"Results saved to {res_path}")
        
        df_path = os.path.join(save_path, f"df_with_scores.json")
        df.to_json(df_path, orient = 'records', indent = 4)
        print(f"DataFrame saved to {df_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Path to model or HuggingFace name", default = 'openai-community/gpt2')
    # parser.add_argument("--bigger_objective", help="whether past or future embeddings should be bigger in norm", default = 'future')
    parser.add_argument("--activations", default='real', type=str, help="`real` or `non-negative`")
    # parser.add_argument("--type", help="type of sample to use", default = 'real')
    args = parser.parse_args()
    main(args)