import torch
from transformers import AutoModel, AutoTokenizer
from utils_data import ProcTextDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from utils_data import Collator
from utils_metrics import get_auc
from utils_topology import apply_step_order
import networkx as nx
import json
import numpy as np
import random
from tqdm.auto import tqdm

json_path_list = [
    './data/erfgc/bio/train.json',
    './data/erfgc/bio/val.json',
    './data/erfgc/bio/test.json',
    ]
data = []
for json_path in json_path_list:
    with open(json_path, 'r', encoding='utf8') as f:
        data += json.load(f)

# shuffle_type = 'unshuffled'
# shuffle_type = 'shuffled'
shuffle_type = 'random_topo'
batch_size = 1

model_name = 'openai-community/gpt2'
# model_name = 'google-bert/bert-base-uncased'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_prefix_space=True,
                                          )
do_add_eos, do_add_bos = False, False
if model_name == 'openai-community/gpt2':
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
                        do_add_eos=do_add_bos,
                        disable_tqdm=disable_tqdm,
                        )
dataset.filter_non_dags()
dataset.filter_short_dags(k = 3)
collator = Collator(tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.dag_collate)
optimizer = AdamW(params=model.parameters(), lr = 5e-5)

num_steps = len(dataset)

auc_list = []
for _ in tqdm(range(100)):
    for batch in tqdm(train_loader, disable = disable_tqdm):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        H_steps_batched = []
        for i in range(batch['input_ids'].size(0)):
            G = batch['G_tokens'][i]
            step_indices = batch['step_indices_tokens'][i]
            num_steps = int(step_indices.max().item()) + 1
            step_order = list(range(1, num_steps)) # NOTE: ignore step 0: that's where BOS/EOS/PAD go
            
            if shuffle_type == 'shuffled':
                step_order = sorted(step_order, key=lambda k: random.random())
            elif shuffle_type == 'random_topo':
                topo_orders = list(nx.all_topological_sorts(G)) 
                if step_order in topo_orders:
                    topo_orders.remove(step_order)
                if len(topo_orders) < 1:
                    continue
                random.shuffle(topo_orders)
                step_order = topo_orders[0]
            input_ids = apply_step_order(input_ids[i], step_order, step_indices)
            
            model_output = model(input_ids = input_ids, attention_mask = attention_mask[i])
            lhs = model_output.last_hidden_state

            h_step_list = []

            A = nx.to_numpy_array(G, nodelist=step_order)   
            
            for j in step_order:
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

mean_auc = sum(auc_list) / len(auc_list)
print(shuffle_type, 'mean_auc:', mean_auc)

    # if len(auc_list) > 0:
    #     mean_auc = sum(auc_list) / len(auc_list)
    #     loss = torch.tensor(mean_auc, requires_grad=True)
    #     # import pdb;pdb.set_trace()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(loss)