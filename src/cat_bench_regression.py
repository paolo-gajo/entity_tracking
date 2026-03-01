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

def get_step_embeddings(batch_df, tokenizer, model, device):
    """
    Concatenates steps into a single sequence, runs the model, 
    and extracts average embeddings for the specific steps asked about.
    """
    inputs = []
    step_spans_batch = []
    
    # Reset index to ensure enumeration matches list indices 0..batch_size
    batch_df = batch_df.reset_index(drop=True)
    
    # 1. Prepare Text and Track Token Spans
    for _, row in batch_df.iterrows():
        steps = row['steps']
        
        # Join with space to mimic natural language flow
        sep = " " 
        
        ids_list = []
        step_ranges = [] # (start_idx, end_idx)
        
        current_len = 0
        
        # Add BOS token if model expects it (optional, but good for GPT-2/RoBERTa)
        # if tokenizer.bos_token_id is not None:
        #     ids_list.append(tokenizer.bos_token_id)
        #     current_len += 1
        for step in steps:
            # Add separator prefix if it's not the start
            prefix = sep if current_len > (1 if tokenizer.bos_token_id else 0) else ""
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
    # Truncate to model max len (e.g. 1024 for GPT2) to prevent OOM/Crash
    model_max = getattr(model.config, 'n_positions', 1024)
    max_len = min(max_len, model_max)
    
    input_ids = torch.zeros((len(inputs), max_len), dtype=torch.long).to(device)
    # Pad with pad_token_id if available, else 0
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
    
    # Use the last hidden state: (Batch, Seq, Hidden)
    # Note: For some models like BERT, you might prefer the second-to-last layer, 
    # but last layer is standard for probing.
    hidden_states = outputs.last_hidden_state
    
    # 4. Extract Specific Step Embeddings
    features = []
    
    # Iterate using simple integer index to match hidden_states[i]
    for i in range(len(batch_df)):
        row_data = batch_df.iloc[i]
        idx_a, idx_b = row_data['step_pair_idx_asked_about']
        
        ranges = step_spans_batch[i]
        
        # Helper to safely extract and pool embedding
        def get_pooling(step_idx, h_state, ranges):
            # Check if step index is valid
            if step_idx >= len(ranges): 
                return torch.zeros(h_state.shape[-1]).to(device)
            
            start, end = ranges[step_idx]
            
            # Handle truncation (if the step was cut off by max_len)
            if start >= h_state.shape[0]: 
                return torch.zeros(h_state.shape[-1]).to(device)
            
            end = min(end, h_state.shape[0])
            
            if start >= end: # Empty span due to truncation
                return torch.zeros(h_state.shape[-1]).to(device)
            
            # Mean pooling over the step tokens
            step_emb = h_state[start:end].mean(dim=0)
            return step_emb

        emb_a = get_pooling(idx_a, hidden_states[i], ranges)
        emb_b = get_pooling(idx_b, hidden_states[i], ranges)

        # Feature: [A, B, |A-B|, A*B] 
        # Note: |A-B| (Absolute diff) is often better for NLI than A-B, but A-B captures direction.
        # Given "Must happen before", direction matters, so A-B is correct.
        feat = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=0)
        features.append(feat.cpu().numpy())
        
    return np.array(features)

def load_and_extract(path, tokenizer, model, device, sample_type='real', batch_size=8, sample_limit=None):
    # Load JSON
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)
    
    # filtering
    if sample_type != 'all':
        df = df[df['type'] == sample_type]
    
    if sample_limit:
        df = df.head(sample_limit)
        
    print(f"Extracting features for {path} ({len(df)} samples)...")
    
    all_feats = []
    all_labels = df['label'].values
    
    # Batch processing
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i+batch_size]
        feats = get_step_embeddings(batch, tokenizer, model, device)
        if len(feats) > 0:
            all_feats.append(feats)
            
    if not all_feats:
        return np.array([]), np.array([])
        
    return np.concatenate(all_feats, axis=0), all_labels

# -------------------------
# Saving
# -------------------------

def get_model_info(model_path, args, task_name="cat_bench_regression"):
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

    for m in model_list:
        model_name = m["path"]
        save_path, train_config = get_model_info(model_name, args)
        result_file = os.path.join(save_path, "results.json")
        
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        print(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Extract features
        print("--- Processing Train Data ---")
        X_train, y_train = load_and_extract(data_path_train, tokenizer, model, device, sample_type=args.sample_type)
        print("--- Processing Test Data ---")
        X_test, y_test = load_and_extract(data_path_test, tokenizer, model, device, sample_type=args.sample_type)

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
    parser = argparse.ArgumentParser(description="Use model embeddings as step embeddings for eval on CaT-Bench")
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--sample_type", default="real", choices=["real", "all"], help="'real' for only real samples, 'all' for real+switched")
    args = parser.parse_args()
    main(args)