from torch.utils.data import Dataset
from typing import List, Dict
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import torch
import networkx as nx

def simple_pad(t, pad_element, pad_length=0):
    padded_t = t + [pad_element] * max(0, (pad_length - len(t)))
    return padded_t

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.grapher = Grapher(tokenizer)

    def dag_collate(self, batch):
        assert len(batch) > 0
        max_len = max([len(el['encodings']['input_ids']) for el in batch])
        input_ids_list = []
        attention_mask_list = []
        step_indices_list = []
        step_indices_tokens_list = []
        head_indices_list = []
        head_indices_tokens_list = []
        batch_edge_list = []
        batch_edge_list_tokens = []
        for el in batch:
            input_ids = el['encodings']['input_ids']
            input_ids_padded = simple_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask = el['encodings']['attention_mask']
            attention_mask_padded = simple_pad(attention_mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
            step_indices = el['step_indices_tokens']
            step_indices_padded = simple_pad(step_indices, pad_element=0, pad_length=max_len)
            step_indices_list.append(torch.tensor(step_indices_padded))
            step_indices_tokens = el['step_indices_tokens']
            step_indices_tokens_padded = simple_pad(step_indices_tokens, pad_element=0, pad_length=max_len)
            step_indices_tokens_list.append(torch.tensor(step_indices_tokens_padded))
            head_indices = el['head_indices_tokens']
            head_indices_padded = simple_pad(head_indices, pad_element=0, pad_length=max_len)
            head_indices_list.append(torch.tensor(head_indices_padded))
            head_indices_tokens = el['head_indices_tokens']
            head_indices_padded_tokens = simple_pad(head_indices_tokens, pad_element=0, pad_length=max_len)
            head_indices_tokens_list.append(torch.tensor(head_indices_padded_tokens))
            edges_words = self.grapher.graph_from_erfgc(
                el['head_indices'],
                el['step_indices'],
                )
            batch_edge_list.append(edges_words)
            edges_tokens = self.grapher.graph_from_erfgc(
                el['head_indices_tokens'],
                el['step_indices_tokens'],
                )
            batch_edge_list_tokens.append(edges_tokens)
            assert edges_words.edges == edges_tokens.edges, 'word- and token-wise edges are not identical'
            print('-' * 100)
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        step_indices_tensor = torch.stack(step_indices_list)
        head_indices_tensor = torch.stack(head_indices_list)
        step_indices_tensor_tokens = torch.stack(step_indices_tokens_list)
        head_indices_tensor_tokens = torch.stack(head_indices_tokens_list)
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'step_indices': step_indices_tensor,
            'head_indices': head_indices_tensor,
            'step_indices_tokens': step_indices_tensor_tokens,
            'head_indices_tokens': head_indices_tensor_tokens,
            'edges': batch_edge_list,
            'edges_tokens': batch_edge_list_tokens,
        }

class ListOfDictsDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
    
    def get_head_indices_tokens(self, word_ids, word_edges):
        T = len(word_ids)
        # map word_id -> first token idx
        word_to_first_token_map = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_to_first_token_map:
                word_to_first_token_map[word_idx] = token_idx
        
        token_edges = [0] * T
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                token_edges[token_idx] = 0
                continue
            target_word_idx = word_edges[word_idx]
            token_edges[token_idx] = word_to_first_token_map.get(target_word_idx, 0)

        return token_edges

    def tokenize(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['encodings'] = self.tokenizer(self.data[i]['words'], is_split_into_words = True)
            word_ids = self.data[i]['encodings'].word_ids()
            self.data[i]['step_indices_tokens'] = [0 if word_idx is None else self.data[i]['step_indices'][word_idx] for word_idx in word_ids]
            self.data[i]['head_indices_tokens'] = self.get_head_indices_tokens(word_ids, self.data[i]['head_indices'])
    
    def add_eos(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['encodings']['input_ids'] = self.data[i]['encodings']['input_ids'] + [self.tokenizer.eos_token_id]
            self.data[i]['encodings']['attention_mask'] = self.data[i]['encodings']['attention_mask'] + [1]
            self.data[i]['step_indices_tokens'] = self.data[i]['step_indices_tokens'] + [0]
            self.data[i]['head_indices_tokens'] = self.data[i]['head_indices_tokens'] + [0]

    def add_bos(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['encodings']['input_ids'] = (
                [self.tokenizer.bos_token_id] +
                self.data[i]['encodings']['input_ids']
                )
            self.data[i]['encodings']['attention_mask'] = [1] + self.data[i]['encodings']['attention_mask']
            self.data[i]['step_indices_tokens'] = [1] + self.data[i]['step_indices_tokens']
            shifted_head_indices_tokens = [0 if el == 0 else el + 1 for el in self.data[i]['head_indices_tokens']]
            self.data[i]['head_indices_tokens'] = [0] + shifted_head_indices_tokens

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class Grapher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def make_edge_list(self, head_indices, step_indices):
        head_indices = [0] + head_indices
        step_indices = [0] + step_indices
        edges = []
        for i, j in enumerate(head_indices):
            src = step_indices[i]
            tgt = step_indices[j]
            if src != tgt and tgt != 0:
                edges.append((src, tgt))
        return edges

    def graph_from_erfgc(self, head_indices, step_indices):
        G = nx.DiGraph()
        edges = self.make_edge_list(head_indices=head_indices, step_indices=step_indices)
        G.add_edges_from(edges)
        unique_steps = set(step_indices) - {0}
        G.add_nodes_from(unique_steps)
        print(G.edges)
        return G