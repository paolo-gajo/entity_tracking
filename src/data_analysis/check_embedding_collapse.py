"""
Diagnose whether MML training causes representation collapse.

For each checkpoint, loads the model, runs it on a sample of RecipeNLG data,
extracts step embeddings, and reports:
  1. Effective rank (via singular value entropy)
  2. Mean pairwise cosine similarity between step embeddings
  3. Std of embedding norms
  4. Fraction of variance in top-k principal components

Usage:
    python src/check_embedding_collapse.py \
        --model_dir models/recipenlg/.../SmolLM2-135M \
        --data_path data/recipenlg/recipenlg_steps_filtered.json \
        --n_samples 200
"""

import argparse
import json
import os
import glob
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(device)
    model.eval()
    return model, tokenizer


def get_step_embeddings_from_sample(steps, tokenizer, model, device):
    """Tokenize steps, run model, mean-pool per step, return [N_steps, D]."""
    ids_list = []
    step_ranges = []
    current_len = 0

    for i, step in enumerate(steps):
        prefix = " " if i > 0 else ""
        step_ids = tokenizer.encode(prefix + step, add_special_tokens=False)
        start = current_len
        end = current_len + len(step_ids)
        step_ranges.append((start, end))
        ids_list.extend(step_ids)
        current_len = end

    model_max = getattr(model.config, 'n_positions',
                        getattr(model.config, 'max_position_embeddings', 1024))
    if len(ids_list) > model_max:
        ids_list = ids_list[:model_max]

    input_ids = torch.tensor([ids_list], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hidden = out.hidden_states[-1][0].float()  # [T, D]

    embs = []
    for start, end in step_ranges:
        end = min(end, hidden.shape[0])
        if start >= end:
            continue
        embs.append(hidden[start:end].mean(dim=0))

    if len(embs) < 2:
        return None
    return torch.stack(embs)  # [S, D]


def effective_rank(S):
    """Effective rank from singular values (Roy & Vetterli, 2007)."""
    p = S / S.sum()
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    return np.exp(entropy)


def analyze_checkpoint(model_path, data, device, n_samples=200):
    model, tokenizer = load_model(model_path, device)

    all_embs = []
    count = 0
    for item in data:
        if count >= n_samples:
            break
        steps = item if isinstance(item, list) else item.get('directions', item.get('orig', None))
        if steps is None or len(steps) < 3:
            continue
        emb = get_step_embeddings_from_sample(steps, tokenizer, model, device)
        if emb is not None:
            all_embs.append(emb.cpu())
            count += 1

    if not all_embs:
        return None

    # Flatten all step embeddings into [total_steps, D]
    flat = torch.cat(all_embs, dim=0).numpy()
    n, d = flat.shape

    # 1. Effective rank
    centered = flat - flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    S_tensor = torch.tensor(S, dtype=torch.float32)
    eff_rank = effective_rank(S_tensor)

    # 2. Variance explained by top-k components
    var_explained = np.cumsum(S ** 2) / (S ** 2).sum()
    top1 = var_explained[0]
    top5 = var_explained[min(4, len(var_explained) - 1)]
    top10 = var_explained[min(9, len(var_explained) - 1)]

    # 3. Mean pairwise cosine similarity (sample 1000 pairs)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / (norms + 1e-8)
    n_pairs = min(2000, n * (n - 1) // 2)
    idx_a = np.random.randint(0, n, n_pairs)
    idx_b = np.random.randint(0, n, n_pairs)
    cos_sims = (normed[idx_a] * normed[idx_b]).sum(axis=1)
    mean_cos = cos_sims.mean()

    # 4. Norm statistics
    norms_flat = norms.flatten()
    mean_norm = norms_flat.mean()
    std_norm = norms_flat.std()
    cv_norm = std_norm / (mean_norm + 1e-8)  # coefficient of variation

    # 5. Per-sample: mean consecutive cosine similarity
    consec_cos = []
    for emb in all_embs:
        e = emb.numpy()
        n_e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        cos = (n_e[:-1] * n_e[1:]).sum(axis=1)
        consec_cos.extend(cos.tolist())
    mean_consec_cos = np.mean(consec_cos) if consec_cos else 0.0

    del model
    torch.cuda.empty_cache()

    return {
        'n_embeddings': n,
        'dim': d,
        'effective_rank': round(eff_rank, 2),
        'var_top1': round(float(top1), 4),
        'var_top5': round(float(top5), 4),
        'var_top10': round(float(top10), 4),
        'mean_pairwise_cos': round(float(mean_cos), 4),
        'mean_consecutive_cos': round(float(mean_consec_cos), 4),
        'mean_norm': round(float(mean_norm), 2),
        'std_norm': round(float(std_norm), 4),
        'cv_norm': round(float(cv_norm), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--base_model', default=None,
                        help='HF model id for baseline (e.g. HuggingFaceTB/SmolLM2-135M). '
                             'If not given, auto-detected from the first checkpoint train_config.json.')
    parser.add_argument('--data_path', default='data/recipenlg/recipenlg_clean.json')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--checkpoints', nargs='*', default=None,
                        help='Specific checkpoint steps to analyze (e.g. 1000 5000 15000 30000). '
                             'If not given, samples ~6 evenly spaced checkpoints.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    # Find checkpoints
    subdirs = sorted(
        [d for d in os.listdir(args.model_dir)
         if os.path.isdir(os.path.join(args.model_dir, d)) and d.isdigit()],
        key=lambda x: int(x)
    )

    if args.checkpoints:
        selected = [s for s in subdirs if s in [str(c) for c in args.checkpoints]]
    else:
        # Sample ~6 evenly spaced
        if len(subdirs) <= 6:
            selected = subdirs
        else:
            indices = np.linspace(0, len(subdirs) - 1, 6, dtype=int)
            selected = [subdirs[i] for i in indices]

    # Auto-detect base model from first checkpoint's train_config.json
    base_model = args.base_model
    if base_model is None:
        for ckpt in subdirs:
            cfg_path = os.path.join(args.model_dir, ckpt, 'train_config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    base_model = json.load(f).get('model_name')
                if base_model:
                    break
    if base_model is None:
        print("WARNING: could not detect base model. Skipping baseline.")

    # Analyze baseline first
    all_labels = []  # (display_name, ckpt_key)
    results = {}

    if base_model:
        print(f"\n--- Baseline: {base_model} ---")
        stats = analyze_checkpoint(base_model, data, device, args.n_samples)
        if stats:
            results['baseline'] = stats
            all_labels.append(('baseline', 'baseline'))
            for k, v in stats.items():
                print(f"  {k:25s}: {v}")

    print(f"\nAnalyzing checkpoints: {selected}")
    print("=" * 100)

    for ckpt in selected:
        path = os.path.join(args.model_dir, ckpt)
        print(f"\n--- Checkpoint {ckpt} ---")
        stats = analyze_checkpoint(path, data, device, args.n_samples)
        if stats:
            results[ckpt] = stats
            all_labels.append((ckpt, ckpt))
            for k, v in stats.items():
                print(f"  {k:25s}: {v}")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'step':>8s} | {'eff_rank':>9s} | {'var_top1':>8s} | {'var_top5':>8s} | "
          f"{'mean_cos':>9s} | {'consec_cos':>10s} | {'mean_norm':>9s} | {'cv_norm':>8s}")
    print("-" * 100)
    for display, key in all_labels:
        if key in results:
            r = results[key]
            print(f"{display:>8s} | {r['effective_rank']:9.2f} | {r['var_top1']:8.4f} | "
                  f"{r['var_top5']:8.4f} | {r['mean_pairwise_cos']:9.4f} | "
                  f"{r['mean_consecutive_cos']:10.4f} | {r['mean_norm']:9.2f} | {r['cv_norm']:8.4f}")


if __name__ == '__main__':
    main()
