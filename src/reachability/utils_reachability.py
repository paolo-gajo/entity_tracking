# src/reachability/utils_reachability.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from scipy import stats
import json
import networkx as nx
from transformers import AutoModel, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import os

from utils.utils_data import ProcTextDataset, Collator
from utils.utils_viz import plot_tensor_heatmap
from utils.utils_sys import setup_config

from data_analysis.pca_utils import run_and_plot_pca

# -------------------------
# Metrics
# -------------------------

def get_auc(S: torch.Tensor, A: np.ndarray) -> float:
    """
    ROC–AUC between continuous scores S and binary adjacency A.
    Excludes diagonal.
    """
    S_np = S.detach().cpu().to(torch.float16).numpy()
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

def get_random_lhs(model, input_ids, step_indices, activations='real'):
    seq_len = input_ids.shape[0]
    hidden_size = model.config.hidden_size
    device = input_ids.device
    lhs = torch.zeros(seq_len, hidden_size, device=device)
    unique_steps = step_indices.unique().tolist()
    for s in unique_steps:
        mask = (step_indices == s)
        v = torch.randn(hidden_size, device=device)
        lhs[mask] = v
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
    dataset.add_num_topos()

    max_steps = max([max(el['step_indices']) for el in dataset])
    h_steps_dict = {k+1: [] for k in range(max_steps)}
    
    collator = Collator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator.dag_collate)

    sources = ['real',
            #    'random',
               ]
    modes = ['directed', 'undirected', 'directed_raw', 'undirected_raw']
    results = {src: {m: {} for m in modes} for src in sources}
    shuffle_types = [
        'unshuffled',
        # 'topological',
        # 'permutations',
        ]

    for shuffle_type in shuffle_types:
        run_means = {src: {m: [] for m in modes} for src in sources}
        n_runs = args.n_runs if shuffle_type != 'unshuffled' else 1

        print(f"Running N={n_runs} random runs for setting={shuffle_type}")
        for run_idx in tqdm(range(n_runs)):
            cur = {src: {m: [] for m in modes} for src in sources}

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
                assert so == nodes, (f"node mismatch: missing={sorted(nodes - so)[:10]}"
                                     f"extra={sorted(so - nodes)[:10]}")

                steps_in_seq = set(orig_indices.unique().tolist()) - {0}
                assert set(step_order) == steps_in_seq

                input_ids, step_indices = apply_step_order(orig_input_ids, step_order, orig_indices)

                A_gold = gold_reachability_matrix(G, step_order)
                with torch.no_grad():
                    lhs_map = {
                        'real': run_model(model, input_ids, batch['attention_mask'][0], args.activations),
                        'random': get_random_lhs(model, input_ids, step_indices, args.activations),
                    }
                
                for src in sources:
                    lhs = lhs_map[src]
                    S_dir, S_undir, H_steps = compute_scores(lhs, step_indices, step_order)

                    # if run_idx == 0 and batch_idx == 0 and args.save_heatmaps:
                    #     base = (model.config.name_or_path
                    #                 if 'models' in model.config.name_or_path 
                    #                 else os.path.join('models', 'baseline', 'gpt2'))
                    #     base = os.path.join(base, src)
                    #     os.makedirs(base, exist_ok=True)
                    #     plot_tensor_heatmap(S_dir,                                          
                    #                       f"./figs/heatmaps/S_directed_{shuffle_type}.pdf")
                    #     plot_tensor_heatmap(S_undir,                                          
                    #                       f"./figs/heatmaps/S_undirected_{shuffle_type}.pdf")

                    R_dir = widest_path_closure(S_dir)
                    R_undir = widest_path_closure(S_undir)

                    A_eval = A_gold.T if args.use_gold_transpose else A_gold
                    auc_d = get_auc(R_dir, A_eval)
                    auc_u = get_auc(R_undir, A_eval)
                    auc_d_raw = get_auc(S_dir, A_eval)
                    auc_u_raw = get_auc(S_undir, A_eval)
                    
                    for pos, step_idx in enumerate(step_order):
                        h_steps_dict[step_idx].append(H_steps[pos])
                    # import pdb; pdb.set_trace()

                    if not np.isnan(auc_d): cur[src]['directed'].append(auc_d)
                    if not np.isnan(auc_u): cur[src]['undirected'].append(auc_u)
                    if not np.isnan(auc_d_raw): cur[src]['directed_raw'].append(auc_d_raw)
                    if not np.isnan(auc_u_raw): cur[src]['undirected_raw'].append(auc_u_raw)

            for src in sources:
                for m in modes:
                    if cur[src][m]:
                        run_means[src][m].append(np.mean(cur[src][m]))
            
            h_steps_dict_stacked = {k: torch.stack(v) for k, v in h_steps_dict.items()}
            
            pre_centroid_dists = np.zeros((max_steps, max_steps))
            post_centroid_dists = np.zeros((max_steps, max_steps))
            if os.path.exists(model_name):
                figs_save_path = model_name
            else:
                figs_save_path = os.path.join('models', 'baseline', model_name)
                os.makedirs(figs_save_path, exist_ok = True)

            figs_save_path = os.path.join(figs_save_path, 'pca')
            if os.path.exists(figs_save_path):
                for file_path in os.listdir(figs_save_path):
                    file_path_abs = os.path.join(figs_save_path, file_path)
                    os.remove(file_path_abs)
            else:
                os.makedirs(figs_save_path)
            for i in tqdm(range(max_steps)):
                for j in range(i+1, max_steps):
                    filename_pca = os.path.join(figs_save_path, f'pca_{i+1}_{j+1}.pdf')
                    centroids = run_and_plot_pca(filename_pca, h_steps_dict_stacked[i+1], h_steps_dict_stacked[j+1])
                    pre_centroid_dists[i, j] = np.linalg.norm(centroids['pre_pca_g1'] - centroids['pre_pca_g2'])
                    post_centroid_dists[i, j] = np.linalg.norm(centroids['post_pca_g1'] - centroids['post_pca_g2'])
            filename_centroid_pre = os.path.join(figs_save_path, 'pre_cent_distances.pdf')
            plot_tensor_heatmap(pre_centroid_dists, filename = filename_centroid_pre)
            filename_centroid_post = os.path.join(figs_save_path, 'post_cent_distances.pdf')
            plot_tensor_heatmap(post_centroid_dists, filename = filename_centroid_post)
            # import pdb; pdb.set_trace()

        for src in sources:
            for mode in modes:
                mu, moe, _ = calculate_statistics(run_means[src][mode])
                vals = run_means[src][mode]
                mn = float(np.min(vals)) if vals else 0.0
                mx = float(np.max(vals)) if vals else 0.0
                results[src][mode][shuffle_type] = {
                    'mu': mu,
                    'moe': moe,
                    'min': mn,
                    'max': mx,
                    'auc': f'{mu:.3f} ± {moe:.3f} [{mn:.3f}, {mx:.3f}]'
                }

    return results

# -------------------------
# Saving (same logic as your sims.py)
# -------------------------

def get_model_info(model_path, args, task_name="erfgc_reachability"):
    train_conf_path = os.path.join(model_path, "train_config.json")

    if os.path.exists(train_conf_path):
        with open(train_conf_path, "r", encoding="utf8") as f:
            train_config = json.load(f)
        model_save_dir = os.path.normpath(train_config["model_save_dir"])
        num_steps = str(train_config.get("num_steps", 0))
        rel = os.path.relpath(model_save_dir, start=os.path.normpath("./models"))
        save_path = os.path.join("./results", task_name, rel, num_steps)
        return save_path, train_config

    train_config = {"num_steps": 0}
    model_leaf = os.path.basename(os.path.normpath(model_path))
    save_path = os.path.join("./results",
                            task_name,
                            "baseline",
                            model_leaf,
                            f"activations={args.activations}",
                            "0"
                            )
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

