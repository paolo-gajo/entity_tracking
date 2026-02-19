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

def get_geometric_scores(batch_df, tokenizer, model, device, bigger_objective = 'future', activations = 'real'):
    inputs = []
    step_spans_batch = []
    
    # Reset index to ensure safe enumeration
    batch_df = batch_df.reset_index(drop=True)
    
    # 1. Tokenization & Spans
    for _, row in batch_df.iterrows():
        steps = row['steps']
        ids_list = []
        step_ranges = [] 
        current_len = 0
        
        # Consistent with your training which used space delimiters
        for step in steps:
            step_text = " " + step.strip()          # always
            step_ids  = tokenizer.encode(step_text, add_special_tokens=False)

            start = current_len
            end   = current_len + len(step_ids)
            step_ranges.append((start, end))

            ids_list.extend(step_ids)
            current_len = end
            
        inputs.append(torch.tensor(ids_list))
        step_spans_batch.append(step_ranges)

    # 2. Pad & Batch
    if not inputs: return np.array([])
        
    max_len = max(len(x) for x in inputs)
    # Cap at model max length (usually 1024 for GPT-2)
    max_len = min(max_len, getattr(model.config, 'n_positions', 1024))
    
    input_ids = torch.zeros((len(inputs), max_len), dtype=torch.long).to(device)
    # Use 0 if pad token is missing (common in base GPT-2)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids.fill_(pad_id)
    
    attention_mask = torch.zeros((len(inputs), max_len), dtype=torch.long).to(device)
    
    for i, seq in enumerate(inputs):
        l = min(len(seq), max_len)
        input_ids[i, :l] = seq[:l]
        attention_mask[i, :l] = 1

    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    hidden_states = outputs.last_hidden_state
    if activations == 'non-negative':
        hidden_states = torch.abs(hidden_states)
    
    scores = []
    
    for i in range(len(batch_df)):
        row_data = batch_df.iloc[i]
        # CatBench: [u, v] means "Must v happen after u?"
        # u = Past (General), v = Future (Specific)
        ranges = step_spans_batch[i]
        idx_a, idx_b = row_data['step_pair_idx_asked_about']
        start_a, end_a = ranges[idx_a]
        start_b, end_b = ranges[idx_b]
        if end_a > max_len or end_b > max_len:
            scores.append(np.nan)  # or skip
            continue
        def get_pooling(step_idx, h_state, ranges):
            if step_idx >= len(ranges): return torch.zeros(h_state.shape[-1]).to(device)
            start, end = ranges[step_idx]
            if start >= h_state.shape[0]: return torch.zeros(h_state.shape[-1]).to(device)
            end = min(end, h_state.shape[0])
            if start >= end: return torch.zeros(h_state.shape[-1]).to(device)
            return h_state[start:end].mean(dim=0)

        direction = row_data.get('direction', 'after')  # or derive from question_type

        emb_a = get_pooling(idx_a, hidden_states[i], ranges)
        emb_b = get_pooling(idx_b, hidden_states[i], ranges)

        if direction == 'after':
            # “must b happen after a?”
            diff = emb_a - emb_b
        elif direction == 'before':
            # “must b happen before a?”  <=> “must a happen after b?”
            diff = emb_b - emb_a
        else:
            raise ValueError(direction)

        energy = torch.relu(diff).pow(2).sum().item()
        scores.append(-energy) # Higher score (0) = Better Fit
        
    return np.array(scores)

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
        print('direction', df['direction'].value_counts(dropna=False))
        bad = set(df['direction'].dropna().unique()) - {'after','before'}
        assert not bad, bad
        print('question_type', df['question_type'].value_counts().head(20))
        print(f"Types of sample: {df['type'].unique()}")
        # Optional: Filter for specific types if you want to inspect subsets
        # print(f'Only using {args.type} samples!')
        # df = df[df['type'] == args.type]    
        print(f"Evaluating {len(df)} samples...")

        all_scores = []
        
        batch_size = 16
        df['score'] = 0.0
        df['objective'] = args.bigger_objective
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i : i+batch_size]
            scores = get_geometric_scores(batch,
                                        tokenizer,
                                        model,
                                        device,
                                        bigger_objective = args.bigger_objective,
                                        activations = args.activations,
                                        )
            df.loc[i : i+batch_size-1, 'score'] = scores
            if len(scores) > 0:
                all_scores.append(scores)
                
        if not all_scores:
            print("No scores computed.")
            return
        
        df_real = df[df['type']=='real'].copy()
        print(df_real.groupby('label')['score'].describe()[['mean','std','count']])
        for d in ['after','before']:
            sub = df_real[df_real['direction']==d]
            if sub['label'].nunique()==2:
                print(d, roc_auc_score(sub['label'], sub['score']), len(sub))

        print("NaN scores:", df['score'].isna().mean(), df['score'].isna().sum())
        print(df[df['score'].isna()].groupby(['type','direction','label']).size().head(20))

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
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC:  {pr_auc:.4f}")
        print("="*40)
        # print(f"ACC REAL: {acc_real:.4f}")
        print(f"ROC AUC REAL: {roc_auc_real:.4f}")
        print(f"PR AUC REAL:  {pr_auc_real:.4f}")
        print("="*40)
        # print(f"ACC SWITCHED: {acc_switched:.4f}")
        print(f"ROC AUC SWITCHED: {roc_auc_switched:.4f}")
        print(f"PR AUC SWITCHED:  {pr_auc_switched:.4f}")
        print("="*40)

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
    parser.add_argument("--bigger_objective", help="whether past or future embeddings should be bigger in norm", default = 'future')
    parser.add_argument("--activations", default='real', type=str, help="`real` or `non-negative`")
    # parser.add_argument("--type", help="type of sample to use", default = 'real')
    args = parser.parse_args()
    main(args)