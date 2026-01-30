from torch.utils.data import Dataset
from typing import List, Dict
from tqdm.auto import tqdm
import numpy as np
from tqdm.auto import tqdm
import torch
import networkx as nx

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
        step_indices_list_tokens = []
        head_indices_list = []
        head_indices_list_tokens = []
        batch_edge_list = []
        batch_edge_list_tokens = []
        for el in batch:
            input_ids = el['tokens']['input_ids']
            input_ids_padded = simple_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask = el['tokens']['attention_mask']
            attention_mask_padded = simple_pad(attention_mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
            # step_indices = el['step_indices_tokens']
            # step_indices_padded = simple_pad(step_indices, pad_element=0, pad_length=max_len)
            step_indices_list.append(torch.tensor(el['step_indices']))
            step_indices_list_tokens.append(torch.tensor(el['step_indices_tokens']))
            # head_indices = el['head_indices_tokens']
            # head_indices_padded = simple_pad(head_indices, pad_element=0, pad_length=max_len)
            head_indices_list.append(torch.tensor(el['head_indices']))
            head_indices_list_tokens.append(torch.tensor(el['head_indices_tokens']))
            batch_edge_list.append(self.grapher.graph_from_erfgc(el['words'], el['head_indices'], el['step_indices']))
            batch_edge_list_tokens.append(self.grapher.graph_from_erfgc(el['tokens']['input_ids'], el['head_indices_tokens'], el['step_indices_tokens']))
            # debug_heads(el["words"], el["head_indices"], self.tokenizer)
            # print('-' * 100)
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        step_indices_tensor = torch.stack(step_indices_list)
        head_indices_tensor = torch.stack(head_indices_list)
        step_indices_tensor_tokens = torch.stack(step_indices_list_tokens)
        head_indices_tensor_tokens = torch.stack(head_indices_list_tokens)
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
    
    def get_head_indices_tokens(self, word_ids, head_indices_word):
        # word_ids: 0-based word id or None
        # head_indices_word: indexed by word id (0..N), values are word ids (0..N), with 0=root
        T = len(word_ids)

        # map word_id -> first token idx
        first_tok = {}
        for tok_i, wid in enumerate(word_ids):
            if wid is None:
                continue
            first_tok.setdefault(wid, tok_i)

        head_tok = np.zeros(T, dtype=np.int64)

        for tok_i, wid in enumerate(word_ids):
            if wid is None:
                head_tok[tok_i] = 0
                continue
            head_wid = head_indices_word[wid]          # 0..N
            head_tok[tok_i] = first_tok.get(head_wid, 0)

        return head_tok

    def tokenize(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['words'] = self.data[i]['words']
            self.data[i]['step_indices'] = self.data[i]['step_indices'] # length N, values 0..N
            N = len(self.data[i]['words']) - 1
            assert max(self.data[i]['head_indices']) <= N, (max(self.data[i]['head_indices']), N)
            self.data[i]['head_indices'] = self.data[i]['head_indices'] # length N
            assert len(self.data[i]['step_indices']) == len(self.data[i]['head_indices'])
            self.data[i]['tokens'] = self.tokenizer(self.data[i]['words'], is_split_into_words = True)
            word_ids = self.data[i]['tokens'].word_ids()
            self.data[i]['step_indices_tokens'] = [0 if wid is None else self.data[i]['step_indices'][wid] for wid in word_ids]
            self.data[i]['head_indices_tokens'] = self.get_head_indices_tokens(word_ids, self.data[i]['head_indices'])
            ...
    
    def add_bos_eos(self):
        for i in tqdm(range(len(self.data))):
            self.data[i]['tokens']['input_ids'] = [self.tokenizer.bos_token_id] + self.data[i]['tokens']['input_ids'] + [self.tokenizer.eos_token_id]
            self.data[i]['tokens']['attention_mask'] = [1] + self.data[i]['tokens']['attention_mask'] + [1]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class Grapher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def make_edge_list(self, head_indices, step_indices):
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
        return G

def debug_heads(words, head_indices_1based, tokenizer):
    """
    words: original words list length N (NO 'root' inserted)
    head_indices_1based: length N, values in 1..N (and optionally 0 if root exists)
    tokenizer: HF tokenizer
    """

    N = len(words)

    # Make explicit word list with root at position 0
    words_r = ["ROOT"] + words

    # Make explicit heads array indexed by word position (0..N)
    # If your dataset has no explicit root heads, this makes them explicit for printing.
    # If some heads are actually root, they should be 0 in head_indices_1based; keep them.
    heads = [0] + list(head_indices_1based)  # heads[wpos] = head_wpos

    # Tokenize with word alignment
    enc = tokenizer(words_r, is_split_into_words=True, add_special_tokens=True)
    toks = enc.tokens()
    wids = enc.word_ids()

    # Map word position -> first token index
    first_tok = {}
    for ti, wid in enumerate(wids):
        if wid is None:
            continue
        first_tok.setdefault(wid, ti)

    # Print word-level head semantics
    print("WORD-LEVEL HEADS (wpos -> head_wpos):")
    for wpos in range(1, N + 1):
        hw = heads[wpos]
        w = words_r[wpos]
        hword = "ROOT" if hw == 0 else words_r[hw]
        print(f"{wpos:>3}: {w:<15} -> {hw:>3} ({hword})")

    print("\nTOKEN-LEVEL MAPPING (word -> first token) and head token chosen:")
    for wpos in range(1, N + 1):
        hw = heads[wpos]
        ti = first_tok.get(wpos, None)
        hti = None if hw == 0 else first_tok.get(hw, None)

        wt = toks[ti] if ti is not None else "<?>"
        ht = "ROOT" if hw == 0 else (toks[hti] if hti is not None else "<?>")

        print(f"{wpos:>3} ({words_r[wpos]:<15}) first_tok={ti:>3} [{wt:<10}]  "
              f"head_wpos={hw:>3} head_tok={('ROOT' if hw==0 else hti):>4} [{ht}]")

    # Basic validity checks
    bad = [(i, h) for i, h in enumerate(heads) if not (0 <= h <= N)]
    if bad:
        print("\nBAD HEAD INDICES (out of range):", bad)
    else:
        print("\nAll head indices are in range 0..N.")
