import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import os
import json
import argparse

def main(args):
    data_path_train = './data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json'
    data_path_test = './data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = args.model_dir
    
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Feature Extraction Helper ---
    def get_step_embeddings(batch_df, tokenizer, model, device):
        """
        Concatenates steps into a single sequence, runs the model, 
        and extracts average embeddings for the specific steps asked about.
        """
        inputs = []
        step_spans_batch = []
        
        # 1. Prepare Text and Track Token Spans
        for _, row in batch_df.iterrows():
            steps = row['steps']
            
            # Simple join with spaces (consistent with pre-training naturalness)
            sep = " " 
            
            # We tokenize strictly chunk-by-chunk to map indices perfectly
            ids_list = []
            step_ranges = [] # (start_idx, end_idx)
            
            current_len = 0
            for step in steps:
                # Add separator prefix if it's not the first step
                prefix = sep if current_len > 0 else ""
                step_text = prefix + step
                
                # Tokenize this chunk
                step_ids = tokenizer.encode(step_text, add_special_tokens=False)
                
                # Record the span
                start = current_len
                end = current_len + len(step_ids)
                step_ranges.append((start, end))
                
                # Extend the actual input list
                ids_list.extend(step_ids)
                current_len = end
                
            inputs.append(torch.tensor(ids_list))
            step_spans_batch.append(step_ranges)

        # 2. Pad and Batch
        if not inputs:
            return np.array([])
            
        max_len = max(len(x) for x in inputs)
        # Truncate to model max len if needed (GPT2 is 1024)
        max_len = min(max_len, 1024)
        
        input_ids = torch.zeros((len(inputs), max_len), dtype=torch.long).to(device)
        attention_mask = torch.zeros((len(inputs), max_len), dtype=torch.long).to(device)
        
        for i, seq in enumerate(inputs):
            l = min(len(seq), max_len)
            input_ids[i, :l] = seq[:l]
            attention_mask[i, :l] = 1

        # 3. Forward Pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Last hidden state: (Batch, Seq, Hidden)
        hidden_states = outputs.last_hidden_state
        
        # 4. Extract Specific Step Embeddings
        features = []
        for i, row in enumerate(batch_df.iterrows()):
            row_data = row[1]
            idx_a, idx_b = row_data['step_pair_idx_asked_about']
            
            ranges = step_spans_batch[i]
            
            # Helper to safely extract and pool embedding
            def get_pooling(idx, h_state, ranges):
                if idx >= len(ranges): 
                    return torch.zeros(h_state.shape[-1]).to(device)
                
                start, end = ranges[idx]
                
                # Handle truncation (if the step was cut off by max_len)
                if start >= h_state.shape[0]: 
                    return torch.zeros(h_state.shape[-1]).to(device)
                
                end = min(end, h_state.shape[0])
                
                # Mean pooling over the step tokens
                step_emb = h_state[start:end].mean(dim=0)
                return step_emb

            emb_a = get_pooling(idx_a, hidden_states[i], ranges)
            emb_b = get_pooling(idx_b, hidden_states[i], ranges)
            
            # Feature: Concatenate [A, B, A-B, A*B] (Standard NLI features)
            feat = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=0)
            features.append(feat.cpu().numpy())
            
        return np.array(features)

    # --- Data Loading Helper ---
    def load_and_extract(path, tokenizer, model, device, batch_size=16, sample_limit=None):
        df = pd.read_json(path)
        df = df[df['type'] == 'real'] # Only use real graph questions
        if sample_limit:
            df = df.head(sample_limit)
            
        print(f"Extracting features for {path} ({len(df)} samples)...")
        
        all_feats = []
        all_labels = df['label'].values
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i : i+batch_size]
            feats = get_step_embeddings(batch, tokenizer, model, device)
            all_feats.append(feats)
            
        return np.concatenate(all_feats, axis=0), all_labels

    # --- Main Execution ---

    # 1. Extract Train Features
    X_train, y_train = load_and_extract(data_path_train, tokenizer, model, device)

    # 2. Extract Test Features
    X_test, y_test = load_and_extract(data_path_test, tokenizer, model, device)

    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # 3. Train Classifier (Linear Probe)
    print("Training Linear Probe (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    report = classification_report(y_test, preds)

    print(f"Model: {model_name}")
    print(report)
    print(f"Macro F1 Score: {f1}")

    # 5. Save Results
    results_dir = './results/cat_bench_probe'
    os.makedirs(results_dir, exist_ok=True)
    res_path = os.path.join(results_dir, f"probe_{model_name.split('/')[-1]}.json")
    with open(res_path, 'w') as f:
        json.dump({'f1': f1, 'report': report}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use model embeddings as step embeddings for eval on CaT-Bench")
    parser.add_argument("--model_dir", help="model dir")
    # "models_tested_kl/recipenlg/natlang/prompt_only_loss/gpt2_14000_kl"
    # openai-community/gpt2
    args = parser.parse_args()
    main(args)