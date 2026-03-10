# sims.py
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
# Metrics
# -------------------------

def get_auc(S: torch.Tensor, A: np.ndarray) -> float:
    """
    ROC–AUC between continuous scores S and binary adjacency A.
    Excludes diagonal.
    """
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

def apply_step_order(t, step_order, step_indices):
    valid_positions = torch.where(step_indices != 0)
    t_shuffled = torch.zeros_like(t)
    buffer_tokens = torch.zeros_like(step_indices[valid_positions])
    buffer_indices = torch.zeros_like(step_indices[valid_positions])
    step_indices_shuffled = torch.zeros_like(step_indices)

    i = 0
    for j in step_order:
        step_positions = torch.where(step_indices == j)[0]
        selected_values = t[step_positions]
        shift = len(selected_values)
        buffer_tokens[i:i+shift] = selected_values
        buffer_indices[i:i+shift] = j
        i += shift

    t_shuffled[valid_positions] = buffer_tokens
    step_indices_shuffled[valid_positions] = buffer_indices
    return t_shuffled, step_indices_shuffled

def get_shuffled_order(G, current_step_indices, shuffle_type):
    num_steps = int(current_step_indices.max().item()) + 1
    step_order = list(range(1, num_steps))  # ignore 0

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
# Model + scoring (same as your sims.py)
# -------------------------

def setup_model(model_name, device):
    print(f'Loading model: {model_name}')
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    if 'gpt2' in model_name:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    return model, tokenizer

def get_step_embeddings(hidden_states, step_indices, step_order):
    h_step_list = []
    for j in step_order:
        mask = torch.where(step_indices == j)[0]
        h_step_indices = hidden_states[mask]
        h_step_pooled = h_step_indices.mean(dim=0)
        h_step_list.append(h_step_pooled)
    return torch.stack(h_step_list)

def run_model(model, input_ids, attention_mask, activations='real'):
    out = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    lhs = out.last_hidden_state.squeeze(0)
    if activations == 'non-negative':
        lhs = torch.abs(lhs)
    return lhs

def compute_scores(hidden_states, step_indices, step_order):
    H_steps = get_step_embeddings(hidden_states, step_indices, step_order)
    # directed: S[i,j] = -||relu(H[i]-H[j])||^2
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)
    penalty = torch.relu(diff).pow(2).sum(dim=-1)
    S_directed = -penalty

    # undirected: cosine
    Hc = H_steps - H_steps.mean(dim=0, keepdim=True)
    Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
    S_undirected = Hn @ Hn.T

    return S_directed, S_undirected, H_steps

# -------------------------
# Reachability from S (continuous)
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
    """
    Gold reachability (transitive closure) adjacency in the same nodelist order.
    """
    G_tc = nx.transitive_closure(G)
    A = nx.to_numpy_array(G_tc, nodelist=step_order).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A

# -------------------------
# Main evaluation loop
# -------------------------

def process_model(model_name, args, data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = setup_model(model_name, device)

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

    collator = Collator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator.dag_collate)

    results = {'directed': {}, 'undirected': {}}
    shuffle_types = ['unshuffled', 'topological', 'permutations']

    for shuffle_type in shuffle_types:
        run_means = {'directed': [], 'undirected': []}
        n_runs = args.n_runs if shuffle_type != 'unshuffled' else 1

        print(f"Processing {shuffle_type} (reachability)...")
        for run_idx in tqdm(range(n_runs), desc="Runs"):
            cur = {'directed': [], 'undirected': []}

            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                G = batch['G_tokens'][0]
                orig_indices = batch['step_indices_tokens'][0]
                orig_input_ids = batch['input_ids'][0]

                step_order = get_shuffled_order(G, orig_indices, shuffle_type)
                if step_order is None:
                    continue

                # consistency checks
                nodes = set(G.nodes()) - {0}
                so = set(step_order)
                assert so == nodes, f"node mismatch: missing={sorted(nodes - so)[:10]} extra={sorted(so - nodes)[:10]}"

                steps_in_seq = set(orig_indices.unique().tolist()) - {0}
                assert set(step_order) == steps_in_seq

                input_ids, step_indices = apply_step_order(orig_input_ids, step_order, orig_indices)

                # Run + score
                lhs = run_model(model, input_ids, batch['attention_mask'][0], args.activations)
                S_dir, S_undir, _ = compute_scores(lhs, step_indices, step_order)

                # Optional plot first example
                if run_idx == 0 and batch_idx == 0 and args.save_heatmaps:
                    base = model.config.name_or_path if 'models' in model.config.name_or_path else os.path.join('models', 'baseline', 'gpt2')
                    os.makedirs(base, exist_ok=True)
                    plot_tensor_heatmap(S_dir, os.path.join(base, f"S_directed_{shuffle_type}.pdf"))
                    plot_tensor_heatmap(S_undir, os.path.join(base, f"S_undirected_{shuffle_type}.pdf"))

                # Gold reachability
                # NOTE: the adjacency matrix order is also permuted.
                # we want to compare A_ij to S_ij, so with a model whose embedding topology
                # perfectly resembles the DAG underlying the recipe
                # we should see that the energy of the node pairs
                # always matches perfectly with the gold reachabilities
                # no matter the permutation of the input into the model
                # i.e. the model's representations 
                A_gold = gold_reachability_matrix(G, step_order)

                # Predicted reachability scores from S
                R_dir = widest_path_closure(S_dir)
                R_undir = widest_path_closure(S_undir)

                # NOTE: keep your historical transpose convention if needed.
                # If your earlier sims needed A.T to align, keep it here too:
                #   auc = get_auc(R_dir, A_gold.T)
                # Otherwise use A_gold.
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
# Saving (same logic as your sims.py)
# -------------------------

def get_model_info(model_path, args, task_name="sims_erfgc_reachability"):
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
        assert len(model_list) == len(set([el['num_steps'] for el in model_list])), "You're os.walking through 2+ model dir trees at once."

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

    args = parser.parse_args()
    main(args)