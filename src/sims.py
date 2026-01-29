import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from utils_data import ListOfDictsDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from utils_data import Collator
from utils_metrics import get_auc
import networkx as nx
import json
import numpy as np

json_path = './data/erfgc/bio/test.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

batch_size = 1

model_name = 'openai-community/gpt2'

tokenizer = GPT2TokenizerFast.from_pretrained(model_name, add_prefix_space=True)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if not tokenizer.bos_token_id:
    tokenizer.bos_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
dataset = ListOfDictsDataset(data, tokenizer)
dataset.tokenize()
# dataset.add_bos_eos()
collator = Collator(tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.dag_collate)
train_iter = iter(train_loader)
optimizer = AdamW(params=model.parameters(), lr = 5e-5)

num_steps = len(dataset)

for step in range(num_steps):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model_output = model.transformer(**batch)
    lhs = model_output.last_hidden_state
    # logits = model.lm_head(lhs)
    
    # print(batch)

    H_steps_batched = []
    auc_list = []
    for i in range(batch['input_ids'].size(0)):
        G = batch['edges'][i]
        # import pdb;pdb.set_trace()
        if not nx.is_directed_acyclic_graph(G):
            # print('Skipped cyclic')
            continue
        if nx.dag_longest_path_length(G) < 2:
            # print('Skipped short')
            continue
        A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        step_indices = batch['step_indices_tokens'][i]
        num_steps = int(step_indices.max().item()) + 1
        h_step_list = []
        for j in range(1, num_steps):
            mask = step_indices == j
            h_step = lhs[i, :][mask].mean(dim = 0)
            h_step_list.append(h_step)
        H_steps = torch.stack(h_step_list)

        # Debiased cosine similarity
        Hc = H_steps - H_steps.mean(dim=0, keepdim=True)
        Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
        S = Hn @ Hn.T  # (n_nodes, n_nodes)

        auc = get_auc(S, A)
        # print(S, A, auc)
        # print('-' * 100)
        if not np.isnan(auc):
            auc_list.append(auc)

        H_steps_batched.append(H_steps)

    # if len(auc_list) > 0:
    #     mean_auc = sum(auc_list) / len(auc_list)
    #     loss = torch.tensor(mean_auc, requires_grad=True)
    #     # import pdb;pdb.set_trace()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(loss)