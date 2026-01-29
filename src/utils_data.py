from torch.utils.data import Dataset
from typing import List, Dict
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import torch
from utils_topology import Grapher

def simple_pad(t, pad_element, pad_length=0):
    return t + [pad_element] * max(0, (pad_length - len(t)))

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.grapher = Grapher(tokenizer)

    def default_collate(self, batch):
        assert len(batch) > 0
        max_len = max([len(el['tokens']['input_ids']) for el in batch])
        input_ids_list = []
        attention_mask_list = []
        for el in batch:
            input_ids = el['tokens']['input_ids']
            input_ids_padded = simple_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask = el['tokens']['attention_mask']
            attention_mask_padded = simple_pad(attention_mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
        }

    def dag_collate(self, batch):
        assert len(batch) > 0
        max_len = max([len(el['tokens']['input_ids']) for el in batch])
        input_ids_list = []
        attention_mask_list = []
        step_indices_list = []
        head_indices_list = []
        batch_edge_list = []
        batch_edge_list_tokens = []
        for el in batch:
            input_ids = el['tokens']['input_ids']
            input_ids_padded = simple_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask = el['tokens']['attention_mask']
            attention_mask_padded = simple_pad(attention_mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
            step_indices = el['step_indices_tokens']
            step_indices_padded = simple_pad(step_indices, pad_element=0, pad_length=max_len)
            step_indices_list.append(torch.tensor(step_indices_padded))
            head_indices = el['head_indices_tokens']
            head_indices_padded = simple_pad(head_indices, pad_element=0, pad_length=max_len)
            head_indices_list.append(torch.tensor(head_indices_padded))
            batch_edge_list.append(self.grapher.graph_from_erfgc(el['words'], el['head_indices'], el['step_indices']))
            batch_edge_list_tokens.append(self.grapher.graph_from_erfgc(el['tokens']['input_ids'], el['head_indices_tokens'], el['step_indices_tokens']))
            print('-' * 100)
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        step_indices_tensor = torch.stack(step_indices_list)
        head_indices_tensor = torch.stack(head_indices_list)
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'step_indices_tokens': step_indices_tensor,
            'head_indices_tokens': head_indices_tensor,
            'edges': batch_edge_list_tokens,
        }

class ListOfDictsDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
    
    def get_head_indices_tokens(self, word_ids, head_indices):
        word_ids_dict = {k: v for v, k in enumerate(word_ids)} # mapping from token position to word position
        head_indices_tokens = [-1 for i in range(len(word_ids))]
        for src, tgt in enumerate(head_indices):
            src_token = word_ids_dict[src]
            tgt_token = word_ids_dict[tgt]
            head_indices_tokens[src_token] = tgt_token
        return head_indices_tokens

    def tokenize(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['tokens'] = self.tokenizer(self.data[i]['words'], is_split_into_words = True)
            word_ids = self.data[i]['tokens'].word_ids()
            self.data[i]['step_indices_tokens'] = [self.data[i]['step_indices'][word_id] for word_id in word_ids]
            self.data[i]['head_indices_tokens'] = self.get_head_indices_tokens(word_ids, self.data[i]['head_indices'])
    def add_bos_eos(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['tokens']['input_ids'] = [self.tokenizer.bos_token_id] + self.data[i]['tokens']['input_ids'] + [self.tokenizer.eos_token_id]
            self.data[i]['tokens']['attention_mask'] = [1] + self.data[i]['tokens']['attention_mask'] + [1]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)