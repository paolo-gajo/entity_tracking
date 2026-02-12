import torch
from transformers import AutoModel, AutoTokenizer
from utils_data import ProcTextDataset, Collator
from torch.utils.data.dataloader import DataLoader
import networkx as nx
import json
import numpy as np
import random
from tqdm.auto import tqdm
import argparse
from scipy import stats
import os
from natsort import natsorted
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# --- Metrics & Math ---

def get_auc(S: torch.Tensor, A: np.ndarray, verbose=False) -> float:
    """
    Compute ROC–AUC between continuous scores S and binary adjacency A.
    Diagonal is excluded.
    """
    S_np = S.detach().cpu().numpy()
    A_np = A.astype(int)

    n = A_np.shape[0]
    # Ensure shapes match
    if S_np.shape != (n, n):
        raise ValueError(f"Shape mismatch: S {S_np.shape} vs A {(n,n)}")

    mask = ~np.eye(n, dtype=bool)
    y_true = A_np[mask]
    y_score = S_np[mask]

    if np.unique(y_true).size < 2:
        return float("nan")
        
    score = roc_auc_score(y_true, y_score)
    if verbose:
        print(f"AUC: {score}")
    return score

def calculate_statistics(auc_list, q=0.975):
    """Compute Mean, SEM, and Margin of Error for a list of AUC scores."""
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
        
    # Handle NaN propagation
    moe = moe if not np.isnan(moe) else 0
    return mu, moe, sem_r

# --- Data & Topology ---

def load_data(json_path_list):
    data = []
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf8') as f:
            data += json.load(f)
    return data

def apply_step_order(t, step_order, step_indices):
    """Reorder tokens in t according to the new step_order."""
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

def get_shuffled_order(G, current_step_indices, shuffle_type, precomputed_topo_orders=None):
    """Determine the order of steps based on the shuffle strategy."""
    num_steps = int(current_step_indices.max().item()) + 1
    step_order = list(range(1, num_steps)) # Ignore step 0 (BOS/EOS/PAD)
    
    if shuffle_type == 'unshuffled':
        return step_order

    # Use precomputed list if available to save time
    if precomputed_topo_orders is not None:
        topo_orders = list(precomputed_topo_orders) # Make a copy to avoid mutation
    else:
        topo_orders = list(nx.all_topological_sorts(G))
    
    if shuffle_type == 'permutations':
        # Random permutation that is NOT a valid topological sort
        step_order_shuffled = step_order
        
        # Check if G has edges (if 0 edges, all perms are valid topo sorts, infinite loop risk)
        if G.number_of_edges() == 0:
             return step_order_shuffled

        # Sample until pi \in P(G) \ T(G)
        while step_order_shuffled in topo_orders and step_order_shuffled == step_order:
            step_order_shuffled = sorted(step_order, key=lambda k: random.random())
        return step_order_shuffled

    elif shuffle_type == 'topological':
        # Random valid topological sort that is NOT the current one
        if step_order in topo_orders:
            topo_orders.remove(step_order)
        if len(topo_orders) < 1:
            return None # Cannot shuffle if only 1 valid order exists
        random.shuffle(topo_orders)
        return topo_orders[0]
    
    return step_order

# --- Model & Inference ---

def setup_model(model_name, device):
    print(f'Loading model: {model_name}')
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    if 'gpt2' in model_name:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            
    return model, tokenizer

def get_step_embeddings(hidden_states, step_indices, step_order):
    """Pool hidden states by step index."""
    h_step_list = []
    for j in step_order:
        mask = torch.where(step_indices == j)[0]
        h_step_indices = hidden_states[mask]
        h_step_pooled = h_step_indices.mean(dim=0)
        h_step_list.append(h_step_pooled)
    return torch.stack(h_step_list)

def compute_scores(model, input_ids, attention_mask, step_indices, step_order):
    """Run model and compute directed and undirected similarity matrices."""
    
    model_output = model(
        input_ids=input_ids.unsqueeze(0), 
        attention_mask=attention_mask.unsqueeze(0)
    )
    lhs = model_output.last_hidden_state.squeeze(0)

    H_steps = get_step_embeddings(lhs, step_indices, step_order)

    # Directed
    diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)
    S_directed = -torch.norm(torch.relu(diff), dim=-1).pow(2)

    # Undirected
    Hc = H_steps - H_steps.mean(dim=0, keepdim=True)
    Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
    S_undirected = Hn @ Hn.T

    return S_directed, S_undirected

# --- Main Logic ---

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

    # Data Structure: run_data[n_topo][mode][shuffle_type] = [list of means per run]
    run_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # NEW: Count unique graphs per bucket
    global_counts = defaultdict(int)
    
    shuffle_types = ['unshuffled', 'topological', 'permutations']

    for shuffle_type in shuffle_types:
        n_runs = args.n_runs if shuffle_type != 'unshuffled' else 1
        
        print(f"Processing {shuffle_type}...")
        for run_idx in tqdm(range(n_runs), desc="Runs"):
            
            # Temporary storage for this run
            current_run_scores = defaultdict(lambda: {'directed': [], 'undirected': []})
            
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                G = batch['G_tokens'][0]
                orig_indices = batch['step_indices_tokens'][0]
                orig_input_ids = batch['input_ids'][0]
                
                # --- NEW: Calculate Complexity (Number of Topo Sorts) ---
                try:
                    topo_orders = list(nx.all_topological_sorts(G))
                    n_topo = len(topo_orders)
                except Exception:
                    continue
                # --------------------------------------------------------

                # Track unique graph counts only during the first run of the first shuffle type
                if shuffle_type == shuffle_types[0] and run_idx == 0:
                    global_counts[n_topo] += 1

                step_order = get_shuffled_order(G, orig_indices, shuffle_type, precomputed_topo_orders=topo_orders)
                if step_order is None: continue 

                input_ids, step_indices = apply_step_order(orig_input_ids, step_order, orig_indices)
                
                S_dir, S_undir = compute_scores(model, input_ids, batch['attention_mask'][0], step_indices, step_order)
                
                A = nx.to_numpy_array(G, nodelist=step_order)
                
                auc_d = get_auc(S_dir, A)
                auc_u = get_auc(S_undir, A)
                
                if not np.isnan(auc_d): current_run_scores[n_topo]['directed'].append(auc_d)
                if not np.isnan(auc_u): current_run_scores[n_topo]['undirected'].append(auc_u)
            
            # Aggregate means for this run
            for n_topo, modes in current_run_scores.items():
                if modes['directed']:
                    mean_val = np.mean(modes['directed'])
                    run_data[n_topo]['directed'][shuffle_type].append(mean_val)
                if modes['undirected']:
                    mean_val = np.mean(modes['undirected'])
                    run_data[n_topo]['undirected'][shuffle_type].append(mean_val)

    # Final Aggregation
    results = {}
    sorted_counts = sorted(run_data.keys())
    total_graphs = sum(global_counts.values())

    for n_topo in sorted_counts:
        results[n_topo] = {
            'meta': {
                'count': global_counts[n_topo],
                'percentage': f"{(global_counts[n_topo] / total_graphs * 100):.2f}%" if total_graphs > 0 else "0%"
            },
            'directed': {}, 
            'undirected': {}
        }
        
        for mode in ['directed', 'undirected']:
            for shuffle_type in shuffle_types:
                data_list = run_data[n_topo][mode].get(shuffle_type, [])
                
                if not data_list:
                    continue
                    
                mu, moe, _ = calculate_statistics(data_list)
                results[n_topo][mode][shuffle_type] = {
                    'mu': mu,
                    'moe': moe,
                    'auc': f'{mu:.3f} $\pm {moe:.3f}$'
                }
            
    return results

def get_model_info(model_path):
    """
    Parses the model path and config to determine the results directory.
    Returns the save path and the loaded train_config (if available).
    """
    model_simple = os.path.basename(os.path.normpath(model_path))
    
    train_conf_path = os.path.join(model_path, 'train_config.json')
    if os.path.exists(train_conf_path):
        with open(train_conf_path, 'r') as f: 
            train_config = json.load(f)
        save_path = os.path.join(
            "./results_by_topo", 
            train_config.get('prompt_type', 'unknown'), 
            train_config.get('loss_type', 'unknown'), 
            model_simple
        )
    else:
        train_config = {}
        save_path = os.path.join('./results_by_topo', "baseline", model_simple)
        
    return save_path, train_config

def save_results_to_disk(results, save_path, train_config, args):
    """
    Saves results to the specified path.
    """
    os.makedirs(save_path, exist_ok=True)
    
    out_dict = {
        'train_config': train_config,
        'eval_config': vars(args), # Convert argparse namespace to dict
        'results': results,
    }
    
    json_path = os.path.join(save_path, "results.json")
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    
    print(f'Results saved to: {json_path}')

def main(args):
    # Load Data
    json_files = [f'./data/erfgc/bio/{split}.json' for split in ['train', 'val', 'test']]
    data = load_data(json_files)

    # Determine Model List
    if not os.path.exists(args.model_dir):
        # Treat as a single model reference (e.g. "openai-community/gpt2")
        model_list = [args.model_dir]
    else:
        # Treat as a directory containing multiple model subdirectories
        model_list = natsorted([
            os.path.join(args.model_dir, el) for el in os.listdir(args.model_dir) 
            if os.path.isdir(os.path.join(args.model_dir, el))
        ])

    for model_name in model_list:
        save_path, train_config = get_model_info(model_name)
        result_file = os.path.join(save_path, "results.json")
        
        if os.path.exists(result_file):
            print(f"Skipping {model_name}: Results already exist at {result_file}")
            continue
        
        results = process_model(model_name, args, data)
        
        if args.verbose_results:
            print(f"Processed {len(results)} complexity groups.")
            # Print distribution summary
            print("\nDistribution Summary:")
            for k, v in results.items():
                print(f"  Topo Orders: {k} | Count: {v['meta']['count']} ({v['meta']['percentage']})")
        
        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--n_runs", default=10, type=int)
    parser.add_argument("--save_results", default=0, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    args = parser.parse_args()
    main(args)