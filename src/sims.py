import torch
from transformers import AutoModel, AutoTokenizer
from utils_data import ProcTextDataset
from torch.utils.data.dataloader import DataLoader
from utils_data import Collator
from utils_metrics import get_auc
from utils_topology import apply_step_order
import networkx as nx
import json
import numpy as np
import random
from tqdm.auto import tqdm
import argparse
from scipy import stats
import os
from natsort import natsorted

def main(args):
    eval_config_dict = args.__dict__
    json_path_list = [
        './data/erfgc/bio/train.json',
        './data/erfgc/bio/val.json',
        './data/erfgc/bio/test.json',
        ]
    data = []
    for json_path in json_path_list:
        with open(json_path, 'r', encoding='utf8') as f:
            data += json.load(f)

    batch_size = 1

    if not os.path.exists(args.model_dir):
        model_list = [args.model_dir]
    else:
        model_list = natsorted([os.path.join(args.model_dir, el) for el in os.listdir(args.model_dir)])
    
    for model_name in model_list:
        print('Model name:', model_name)
        train_config_filename = os.path.join(model_name, 'train_config.json')
        model_name_simple = model_name.split('/')[-1]
        if os.path.exists(train_config_filename):
            with open(train_config_filename, 'r', encoding='utf8') as f:
                train_config_dict = json.load(f)
            prompt_type = train_config_dict['prompt_type']
            loss_type = train_config_dict['loss_type']
            results_save_path = os.path.join("./results", model_name_simple, prompt_type, loss_type)
        else:
            train_config_dict = {}
            results_save_path = os.path.join('./results', model_name_simple, "baseline")

        
        print('Will save results to: ', results_save_path)
        os.makedirs(results_save_path, exist_ok=True)
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                add_prefix_space=True,
                                                )
        do_add_eos, do_add_bos = False, False
        if 'gpt2' in model_name:
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if not tokenizer.bos_token_id:
                tokenizer.bos_token_id = tokenizer.eos_token_id
            do_add_eos, do_add_bos = True, True

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        disable_tqdm = True
        dataset = ProcTextDataset(data,
                                tokenizer,
                                do_tokenize=True,
                                do_add_bos=do_add_bos,
                                do_add_eos=do_add_eos,
                                disable_tqdm=disable_tqdm,
                                )
        dataset.filter_non_dags()
        dataset.filter_short_dags(k = 2)
        collator = Collator(tokenizer)
        results_dict = {
            'unshuffled': {},
            'topological': {},
            'permutations': {},
        }
        for shuffle_type in results_dict:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.dag_collate)
            m_r_list = []
            n_runs = args.n_runs if shuffle_type != 'unshuffled' else 1
            for _ in tqdm(range(n_runs)):
                auc_list = []
                for batch in tqdm(dataloader, disable = disable_tqdm):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    H_steps_batched = []
                    num_elements_batch = batch['input_ids'].size(0)
                    for i in range(num_elements_batch):
                        G = batch['G_tokens'][i]
                        step_indices = batch['step_indices_tokens'][i]
                        num_steps = int(step_indices.max().item()) + 1
                        step_order = list(range(1, num_steps)) # NOTE: ignore step 0: that's where BOS/EOS/PAD go
                        topo_orders = list(nx.all_topological_sorts(G))

                        # if we look at random permutations, we want the order to come from the difference
                        # between all possible permutations and the valid topological orders
                        # and we also want ot exclude the original order
                        if shuffle_type == 'permutations':
                            step_order_shuffled = step_order
                            while step_order_shuffled in topo_orders and step_order_shuffled == step_order: # sample until pi \in P(G) \ T(G)
                                step_order_shuffled = sorted(step_order, key=lambda k: random.random())
                            step_order = step_order_shuffled
                        # here we want to draw just from valid topological orders
                        # but we also remove the original order
                        elif shuffle_type == 'topological': 
                            if step_order in topo_orders:
                                topo_orders.remove(step_order)
                            if len(topo_orders) < 1:
                                continue
                            random.shuffle(topo_orders)
                            step_order = topo_orders[0]

                        input_ids, step_indices = apply_step_order(input_ids[i], step_order, step_indices)

                        model_output = model(input_ids = input_ids.unsqueeze(0).expand(num_elements_batch, -1),
                                            attention_mask = attention_mask,)
                        lhs = model_output.last_hidden_state.squeeze()

                        h_step_list = []

                        A = nx.to_numpy_array(G, nodelist=step_order)   
                        
                        for j in step_order: # 2 3 1 4 5
                            mask = torch.where(step_indices == j)[0]
                            h_step = torch.index_select(lhs, 0, mask).mean(dim = 0)
                            h_step_list.append(h_step)
                        H_steps = torch.stack(h_step_list)

                        # Debiased cosine similarity
                        Hc = H_steps - H_steps.mean(dim=0, keepdim=True)
                        Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
                        S = Hn @ Hn.T  # (n_nodes, n_nodes)

                        auc = get_auc(S, A, verbose=False)
                        if not np.isnan(auc):
                            auc_list.append(auc)
                        H_steps_batched.append(H_steps)

                m_r_list.append(np.mean(auc_list))
            mu = np.mean(m_r_list)
            degs = len(m_r_list) - 1
            sem_r = stats.sem(m_r_list)
            q = 0.975
            t_critical = stats.t.ppf(q, degs)

            # Critical value t_{R-1, 0.975}
            moe = t_critical * sem_r
            moe = moe if not np.isnan(moe) else 0
            results_dict[shuffle_type] = {
                'model_dir': model_name,
                'shuffle_type': shuffle_type,
                'n_runs': n_runs,
                'mu': mu,
                'moe': moe if not np.isnan(moe) else 0,
                'auc': f'{mu:.3f} $\pm {moe:.3f}$',
                'q': q,
                
            }
        print(results_dict)

        results_dict_json_path = os.path.join(results_save_path, "results.json")
        print('Results saved to:', results_dict_json_path)
        
        out_dict = {
            'train_config': train_config_dict,
            'eval_config': eval_config_dict,
            'results': results_dict,
        }
        with open(results_dict_json_path, 'w', encoding='utf8') as f:
            json.dump(out_dict, f, ensure_ascii = False, indent = 4)
        if os.path.exists(model_name):
            os.makedirs(model_name.replace('models', 'models_tested'))
            os.rename(model_name, model_name.replace('models', 'models_tested'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate sims between step embeddings and compare them to ground truth step topology")
    parser.add_argument("--model_dir", help="name or path of the model", default="openai-community/gpt2")
    parser.add_argument("--n_runs", help="number of runs", default=10, type=int)
    args = parser.parse_args()
    main(args)