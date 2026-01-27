import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import permutations
from tqdm.auto import tqdm
from data_utils import graph_from_erfgc, get_ordered_text, batched_gen
import networkx as nx
import json

def get_perplexity(text_batch, model, tokenizer, device):
    # NOTE: even though padding should be useless since
    # we are feeding permutations of the same procedural text
    # (which should all be the same length even when tokenized)
    # sequences are not symmetric under tokenization
    try:
        inputs = tokenizer(text_batch, return_tensors="pt", padding = 'longest').to(device)
    except:
        import pdb; pdb.set_trace()
    
    # Create labels for shifting
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # 1. Forward pass without labels to get logits (no loss calculation yet)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits

    # 2. Shift logits and labels 
    # GPT-2 is causal: logit at index t predicts token at index t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 3. Calculate loss per token (reduction='none' keeps the shape)
    loss_fct = CrossEntropyLoss(reduction='none')
    # Flatten to [batch_size * (seq_len-1), vocab_size] for the loss function
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Reshape back to [batch_size, seq_len-1]
    loss = loss.view(len(text_batch), -1)

    # 4. Average loss per sequence (ignoring padding)
    # Create a mask where labels are valid (not -100)
    valid_mask = (shift_labels != -100).float()
    
    # Sum loss over valid tokens for each sequence
    sum_loss = (loss * valid_mask).sum(dim=1)
    
    # Count valid tokens for each sequence
    num_valid = valid_mask.sum(dim=1)
    
    # Average loss per sequence
    mean_loss = sum_loss / num_valid
    
    # 5. Calculate perplexity per sequence
    perplexities = torch.exp(mean_loss)
    # import pdb; pdb.set_trace()
    return perplexities  # Returns a tensor of shape [batch_size]

def main():
    '''
    We wanna get the min, mean, max perplexity
    for (1) topological orders and (2) all permutations
    of procedural texts, which are lists of lists of strings,
    i.e. lists of word lists.
    '''
    model_id = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    json_path = './data/erfgc/bio/train.json'
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    perplexity_list_topos = []
    perplexity_list_perms = []
    batch_size = 64
    candidate_perms = []
    for data_idx in tqdm(range(len(data))):
        sample = data[data_idx]
        head_indices = np.array([0] + sample['head_indices'])
        words = np.array(['root'] + sample['words'])
        step_indices = np.array([0] + sample['step_indices'])

        G = graph_from_erfgc(head_indices, step_indices)
        N = len(G.nodes)
        nodes = list(G.nodes)
        if N < 4 or N > 7:
            continue
        try:
            topological_sorts = list(nx.all_topological_sorts(G))
        except nx.NetworkXUnfeasible:
            continue

        for order_batch in tqdm(batched_gen(topological_sorts, batch_size)):
            text_batch = get_ordered_text(order_batch, words, step_indices)
            perplexity_list = get_perplexity(text_batch, model, tokenizer, 'cuda')
            perplexity_list_topos.extend(perplexity_list.tolist())
        
        for order_batch in tqdm(batched_gen(permutations(nodes), batch_size)):
            text_batch = get_ordered_text(order_batch, words, step_indices)
            perplexity_list = get_perplexity(text_batch, model, tokenizer, 'cuda')
            perplexity_list_perms.extend(perplexity_list.tolist())
            for text, perp, order in zip(text_batch, perplexity_list, order_batch):
                candidate_perms.append({
                    'data_idx': data_idx,
                    'valid': 1 if order in topological_sorts else 0,
                    'n_nodes': N,
                    'perp': perp.item(),
                    'text': text,
                    'order': [int(el) for el in order],
                    'head_indices': head_indices.tolist(),
                    'words': words.tolist(),
                    'step_indices': step_indices.tolist(),
                })
    
    min_perplexity_topos = min(perplexity_list_topos)
    max_perplexity_topos = max(perplexity_list_topos)
    mean_perplexity_topos = sum(perplexity_list_topos) / len(perplexity_list_topos)
    print(f'Min perplexity topos: {min_perplexity_topos}')
    print(f'Max perplexity topos: {max_perplexity_topos}')
    print(f'Mean perplexity topos: {mean_perplexity_topos}')
    candidate_perms_json_path = 'candidate_perms.json'    
    with open(candidate_perms_json_path, 'w', encoding='utf8') as f:
        json.dump(candidate_perms, f, ensure_ascii = False, indent = 4)

    candidate_perms_filtered = []
    for cand in candidate_perms:
        if cand['perp'] <= min_perplexity_topos:
            candidate_perms_filtered.append(cand)

    candidate_perms_filtered_json_path = 'candidate_perms_filtered.json'    
    with open(candidate_perms_filtered_json_path, 'w', encoding='utf8') as f:
        json.dump(candidate_perms_filtered, f, ensure_ascii = False, indent = 4)

    min_perplexity_perms = min(perplexity_list_perms)
    max_perplexity_perms = max(perplexity_list_perms)
    mean_perplexity_perms = sum(perplexity_list_perms) / len(perplexity_list_perms)
    print(f'Min perplexity perms: {min_perplexity_perms}')
    print(f'Max perplexity perms: {max_perplexity_perms}')
    print(f'Mean perplexity perms: {mean_perplexity_perms}')

if __name__ == "__main__":
    main()