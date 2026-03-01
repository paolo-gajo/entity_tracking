# sims_step_tokens.py
#
# Like sims.py, but uses the hidden state at <step_j> token positions
# instead of mean-pooling content tokens per step.
#
# Each step's text is followed by a <step_j> token (assigned by input order).
# The hidden state at that position is the step embedding.

import torch
from transformers import AutoModel, AutoTokenizer
from utils_data import ProcTextDataset, Collator
from utils_viz import plot_tensor_heatmap
from torch.utils.data.dataloader import DataLoader
import networkx as nx
import json
import numpy as np
import random
from tqdm.auto import tqdm
import argparse
from scipy import stats
import os
from sklearn.metrics import roc_auc_score
from utils_sys import setup_config

# -------------------------
# Metrics (same as sims.py)
# -------------------------

def get_auc(S: torch.Tensor, A: np.ndarray) -> float:
    S_np = S.detach().cpu().numpy()
    A_np = A.astype(int)
    n = A_np.shape[0]
    if S_np.shape != (n, n):
        raise ValueError(f"Shape mismatch: S {S_np.shape} vs A {(n,n)}")
    mask = ~np.eye(n, dtype=bool)
    y_true = A_np[mask]
    y_score = S_np[mask]
    if np.unique(y_true).size < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)

def calculate_statistics(auc_list, q=0.975):
    if not auc_list:
        return 0.0, 0.0, 0.0
    mu = np.mean(auc_list)
    sem_r = stats.sem(auc_list)
    degs = len(auc_list) - 1
    if degs > 0:
        t_critical = stats.t.ppf(q, degs)
        moe = t_critical * sem_r
    else:
        moe = 0.0
    moe = moe if not np.isnan(moe) else 0.0
    return mu, moe, sem_r

# -------------------------
# Data helpers
# -------------------------

def load_data(json_path_list):
    data = []
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf8') as f:
            data += json.load(f)
    return data

def get_shuffled_order(G, current_step_indices, shuffle_type):
    num_steps = int(current_step_indices.max().item()) + 1
    step_order = list(range(1, num_steps))

    if shuffle_type == 'unshuffled':
        return step_order

    topo_orders = list(nx.all_topological_sorts(G))

    if shuffle_type == 'permutations':
        step_order_shuffled = step_order
        while (step_order_shuffled in topo_orders) or (step_order_shuffled == step_order):
            step_order_shuffled = sorted(step_order, key=lambda k: random.random())
        return step_order_shuffled

    if shuffle_type == 'topological':
        if step_order in topo_orders:
            topo_orders.remove(step_order)
        if len(topo_orders) < 1:
            return None
        random.shuffle(topo_orders)
        return topo_orders[0]

    return step_order

# -------------------------
# Build input with step tokens
# -------------------------

def build_step_token_input(words, step_indices_word, step_order, tokenizer, step_token_ids, device):
    """
    Given word-level data and a step_order permutation, build token-level
    input_ids with <step_j> appended after each step's content.

    Step tokens are assigned by input order (j-th step in the sequence gets
    <step_j>), NOT by original step index.

    Returns:
        input_ids:          [T]  token ids including step tokens
        step_token_positions: dict mapping step_index -> position of its <step_*> token
        step_indices_tokens: [T]  step index per token (1-indexed, 0 for non-step)
        attention_mask:      [T]  all ones
    """
    input_ids = []
    step_indices_tokens = []
    step_token_positions = {}  # step_index -> position in input_ids

    for input_order_j, step_idx in enumerate(step_order):
        # Get word positions for this step
        word_positions = [i for i, s in enumerate(step_indices_word) if s == step_idx]
        if not word_positions:
            continue

        # Tokenize the step's words
        step_words = [words[i] for i in word_positions]
        step_text = " " + " ".join(step_words)
        step_tok_ids = tokenizer.encode(step_text, add_special_tokens=False)

        # Content tokens
        input_ids.extend(step_tok_ids)
        step_indices_tokens.extend([step_idx] * len(step_tok_ids))

        # Step token (assigned by input order)
        if input_order_j < len(step_token_ids):
            stp_id = step_token_ids[input_order_j]
        else:
            # More steps than available step tokens — use last one
            stp_id = step_token_ids[-1]

        step_token_positions[step_idx] = len(input_ids)
        input_ids.append(stp_id)
        step_indices_tokens.append(step_idx)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    step_indices_tokens = torch.tensor(step_indices_tokens, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, step_token_positions, step_indices_tokens, attention_mask


def get_step_embeddings_from_positions(hidden_states, step_token_positions, step_order):
    """
    Extract hidden states at step token positions, in the order given by step_order.
    """
    h_list = []
    for step_idx in step_order:
        pos = step_token_positions[step_idx]
        h_list.append(hidden_states[pos])
    return torch.stack(h_list)  # [N, D]

# -------------------------
# Scoring (same as sims.py)
# -------------------------

def compute_scores(H_steps):
    # directed: S[i,j] = -||relu(H[i]-H[j])||^2
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)
    penalty = torch.relu(diff).pow(2).sum(dim=-1)
    S_directed = -penalty

    # undirected: cosine
    Hc = H_steps - H_steps.mean(dim=0, keepdim=True)
    Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
    S_undirected = Hn @ Hn.T

    return S_directed, S_undirected

# -------------------------
# Reachability
# -------------------------

def widest_path_closure(S: torch.Tensor) -> torch.Tensor:
    R = S.clone()
    n = R.shape[0]
    R.fill_diagonal_(-1e9)
    for k in range(n):
        via = torch.minimum(R[:, k].unsqueeze(1), R[k, :].unsqueeze(0))
        R = torch.maximum(R, via)
    R.fill_diagonal_(-1e9)
    return R

def gold_reachability_matrix(G: nx.DiGraph, step_order):
    G_tc = nx.transitive_closure(G)
    A = nx.to_numpy_array(G_tc, nodelist=step_order).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A

# -------------------------
# Model setup
# -------------------------

def setup_model(model_name, device, stp_max_steps=15):
    print(f'Loading model: {model_name}')
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if 'gpt2' in model_name.lower():
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id

    # If the tokenizer doesn't already have step tokens (e.g. baseline),
    # add them and resize embeddings so we get a random-init baseline.
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

def get_step_token_ids(tokenizer, max_steps):
    """Get the token IDs for <step_0>, <step_1>, ..., <step_{max_steps-1}>."""
    ids = []
    for i in range(max_steps):
        tok = f"<step_{i}>"
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid == tokenizer.unk_token_id:
            print(f"WARNING: {tok} not found in tokenizer vocabulary")
            break
        ids.append(tid)
    return ids

# -------------------------
# Main evaluation loop
# -------------------------

def process_model(model_name, args, data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, step_token_ids = setup_model(model_name, device, args.stp_max_steps)

    print(f"Using {len(step_token_ids)} step tokens")

    dataset = ProcTextDataset(
        data,
        tokenizer,
        do_tokenize=True,
        do_add_bos=False,
        do_add_eos=False,
        disable_tqdm=True
    )
    dataset.filter_non_dags()
    dataset.filter_short_dags(k=2)

    results = {'directed': {}, 'undirected': {}}
    shuffle_types = ['unshuffled', 'topological', 'permutations']

    for shuffle_type in shuffle_types:
        run_means = {'directed': [], 'undirected': []}
        n_runs = args.n_runs if shuffle_type != 'unshuffled' else 1

        print(f"Processing {shuffle_type}...")
        for run_idx in tqdm(range(n_runs), desc="Runs"):
            cur = {'directed': [], 'undirected': []}

            for sample in dataset:
                G = sample['G_tokens']
                orig_step_indices = sample['step_indices_tokens']
                orig_step_indices_t = torch.tensor(orig_step_indices, dtype=torch.long)
                words = sample['words']
                word_step_indices = sample['step_indices']

                n_steps = len(set(word_step_indices) - {0})

                # Skip if more steps than step tokens
                if n_steps > len(step_token_ids):
                    continue

                step_order = get_shuffled_order(G, orig_step_indices_t, shuffle_type)
                if step_order is None:
                    continue

                # Build input with step tokens
                input_ids, stp_positions, step_indices_tokens, attn_mask = build_step_token_input(
                    words, word_step_indices, step_order, tokenizer, step_token_ids, device
                )

                if input_ids.numel() == 0:
                    continue

                max_len = getattr(model.config, 'n_positions', 1024)
                if input_ids.numel() > max_len:
                    continue

                # Forward
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attn_mask.unsqueeze(0),
                    )
                lhs = out.last_hidden_state.squeeze(0)  # [T, D]

                if args.activations == 'non-negative':
                    lhs = torch.abs(lhs)

                # Get step embeddings from step token positions
                H_steps = get_step_embeddings_from_positions(lhs, stp_positions, step_order)

                # Score
                S_dir, S_undir = compute_scores(H_steps)

                # Optional heatmaps
                if run_idx == 0 and cur['directed'] == [] and args.save_heatmaps:
                    base = model.config.name_or_path if 'models' in model.config.name_or_path else os.path.join('models', 'baseline', 'gpt2')
                    os.makedirs(base, exist_ok=True)
                    plot_tensor_heatmap(S_dir, os.path.join(base, f"S_directed_stp_{shuffle_type}.pdf"))
                    plot_tensor_heatmap(S_undir, os.path.join(base, f"S_undirected_stp_{shuffle_type}.pdf"))

                # Gold reachability
                A_gold = gold_reachability_matrix(G, step_order)

                # Predicted reachability via widest-path closure
                R_dir = widest_path_closure(S_dir)
                R_undir = widest_path_closure(S_undir)

                if args.use_gold_transpose:
                    auc_d = get_auc(R_dir, A_gold.T)
                    auc_u = get_auc(R_undir, A_gold.T)
                else:
                    auc_d = get_auc(R_dir, A_gold)
                    auc_u = get_auc(R_undir, A_gold)

                if not np.isnan(auc_d): cur['directed'].append(auc_d)
                if not np.isnan(auc_u): cur['undirected'].append(auc_u)

            if cur['directed']:
                run_means['directed'].append(np.mean(cur['directed']))
            if cur['undirected']:
                run_means['undirected'].append(np.mean(cur['undirected']))

        for mode in ['directed', 'undirected']:
            mu, moe, _ = calculate_statistics(run_means[mode])
            results[mode][shuffle_type] = {
                'mu': mu,
                'moe': moe,
                'auc': f'{mu:.3f} ± {moe:.3f}'
            }

    return results

# -------------------------
# Saving
# -------------------------

def get_model_info(model_path, args, task_name="sims_erfgc_step_tokens"):
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
        'train_config': train_config,
        'eval_config': vars(args),
        'results': results,
    }
    json_path = os.path.join(save_path, "results.json")
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {json_path}")

def main(args):
    json_files = [f'./data/erfgc/bio/{split}.json' for split in ['train', 'val', 'test']]
    data = load_data(json_files)

    if not os.path.exists(args.model_dir):
        model_list = [{'path': args.model_dir, 'num_steps': 0}]
    else:
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for F in files:
                if F == 'train_config.json':
                    with open(os.path.join(root, F), 'r', encoding='utf8') as f:
                        num_steps = json.load(f)['num_steps']
                    model_list.append({'path': root, 'num_steps': num_steps})
        model_list = sorted(model_list, key=lambda x: x['num_steps'])
        assert len(model_list) == len(set([el['num_steps'] for el in model_list]))

    for model in model_list:
        model_name = model['path']
        save_path, train_config = get_model_info(model_name, args)
        result_file = os.path.join(save_path, "results.json")
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        results = process_model(model_name, args, data)

        if args.verbose_results:
            print(json.dumps(results, indent=4))

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--activations", default="real", type=str, help="real | non-negative")
    parser.add_argument("--save_heatmaps", default=0, type=int)
    parser.add_argument("--use_gold_transpose", default=0, type=int)
    parser.add_argument("--stp_max_steps", default=15, type=int, help="Number of step tokens to look for in tokenizer")

    args = parser.parse_args()
    main(args)
