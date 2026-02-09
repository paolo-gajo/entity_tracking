import torch
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm.auto import tqdm
from tqdm.auto import tqdm
import networkx as nx
import pandas as pd
import random

def make_pairs_from_recipenlg(data):
    print(f"Dataset Size: {len(data)}")
    step_list_orig = [x['directions'] for x in tqdm(data, desc='Filtering...') if len(set(x['directions'])) > 1]
    
    total_samples = len(step_list_orig)
    halfway_point = total_samples // 2
    
    sample = random.sample
    data_pairs = []

    # 3. Regime A: Forced Negative (Indices 0 to N/2)
    for i in tqdm(range(halfway_point), desc = 'Making negatives...'):
        orig = step_list_orig[i]
        n_steps = len(orig)
        shuf = sample(orig, n_steps)
        # Enforce inequality
        while shuf == orig:
            shuf = sample(orig, n_steps)
            
        data_pairs.append({
            'orig': orig,
            'shuf': shuf,
            'binary_label': int(orig == shuf)
        })

    # 4. Regime B: Identity / Positive (Indices N/2 to N)
    for i in tqdm(range(halfway_point, total_samples), desc = 'Copying neutrals...'):
        orig = step_list_orig[i]
        shuf = orig  # Identity assignment
        
        data_pairs.append({
            'orig': orig,
            'shuf': shuf,
            'binary_label': int(orig == shuf)
        })

    binary_labels_positive = sum(d['binary_label'] for d in data_pairs)
    pos_ratio = binary_labels_positive / len(data_pairs) if data_pairs else 0.0
    
    print(f'Positive/negative sample ratio: {pos_ratio:.6f}')
    return data_pairs

class ICLDataset(Dataset):
    def __init__(self,
                 icl_dataset,
                 test_dataset,
                 tokenizer,
                 n_icl,
                 max_length = 1024,
                 prune_lengths = True,
                 num_samples = 0,
                 ):
        super().__init__()
        self.icl_dataset = icl_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.n_icl = n_icl
        self.max_length = max_length
        self.data = self.test_dataset.apply(lambda x: self.make_icl_sample(x), axis = 1)
        if num_samples > 0:
            self.data = self.data[:num_samples]
        if prune_lengths:
            self.prune_too_long()

    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data)

    def prune_too_long(self):
        self.data = self.data.apply(lambda x: x if not x['input_ids'].numel() > self.max_length else None)
        self.data = self.data.dropna()

    def format_steps(self, line, append_labels = False, labels_nl = ('no', 'yes')):
        steps = line['steps']
        steps_joined = 'Steps: ' + ' '.join(steps)
        idx_head = line['step_pair_idx_asked_about'][0]
        head = f" Step {line['step_pair_idx_asked_about'][0] + 1}: " + steps[idx_head]
        idx_tail = line['step_pair_idx_asked_about'][1]
        tail = f" Step {line['step_pair_idx_asked_about'][1] + 1}: " + steps[idx_tail]
        question = f" {line['binary_question']}"
        label_text = labels_nl[line['label']]
        answer = f" Answer: {label_text}"
        prompt = steps_joined + '\n\n' + head + '\n\n' + tail + '\n\n' + question + '\n\n' + answer + '\n\n'
        prompt_tokens = self.tokenizer(prompt)['input_ids']
        return prompt_tokens
    
    def make_icl_sample(self, line, sample_type = 'real'):
        icl_dataset = self.icl_dataset
        if sample_type:
            icl_dataset = icl_dataset[icl_dataset['type'] == sample_type]
        df_icl = icl_dataset.groupby(['label', 'type',
                                      'direction',
                                      ], group_keys=False).sample(n=self.n_icl, replace=False)
        df_icl = df_icl.sample(frac=1)
        icl_input_ids = df_icl.apply(lambda x: self.format_steps(x, append_labels = True), axis = 1)
        icl_input_ids_tensors = [torch.tensor(el) for el in icl_input_ids.to_list()]
        test_input_ids = self.format_steps(line, append_labels = True)
        test_input_ids_tensor = torch.tensor(test_input_ids)[:-1]
        input_ids = torch.concat(icl_input_ids_tensors + [test_input_ids_tensor])
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids, dtype = input_ids.dtype),
            }

def pad_collate(batch, tokenizer, side = 'right'):
    max_len = max([len(el['input_ids']) for el in batch])
    input_ids_padded_list = []
    attention_mask_padded_list = []
    for el in batch:
        input_ids_padded = tensor_pad(el['input_ids'], tokenizer.pad_token_id, max_len, side = side)
        input_ids_padded_list.append(torch.tensor(input_ids_padded))
        attention_mask_padded = tensor_pad(el['attention_mask'], 0, max_len, side = side)
        attention_mask_padded_list.append(torch.tensor(attention_mask_padded))
    input_ids_padded_tensor = torch.stack(input_ids_padded_list)
    attention_mask_padded_tensor = torch.stack(attention_mask_padded_list)
    return {
        'input_ids': input_ids_padded_tensor,
        'attention_mask': attention_mask_padded_tensor,
    }

def list_pad(t, pad_element, pad_length=0):
    padded_t = t + [pad_element] * max(0, (pad_length - len(t)))
    return padded_t

def tensor_pad(t, pad_element, pad_length=0, side = 'right'):
    def apply_padding(t, t_pad, dim, side=side):
        if side == 'right':
            return torch.cat([t, t_pad], dim = dim)
        else:
            return torch.cat([t_pad, t], dim = dim)
        
    assert t.dim() > 0
    if t.dim() == 1:
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(0)))
        t_pad = t_pad.to(dtype = t.dtype)
        return apply_padding(t, t_pad, dim = 0)
    elif t.dim() == 2:
        B, _ = t.shape
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(1)))
        t_pad = t_pad.to(dtype = t.dtype)
        t_pad = t_pad.expand(B, pad_length)
        return apply_padding(t, t_pad, dim = 1)
    elif t.dim() == 3:
        B, _, D = t.shape
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(1)))
        t_pad = t_pad.to(dtype = t.dtype)
        t_pad = t_pad.expand(B, pad_length, D)
        return apply_padding(t, t_pad, dim = 1)

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def seq2seq_collate(self, batch):
        max_len = max([len(el['input_ids']) for el in batch])
        input_ids_list = []
        attention_mask_list = []
        for el in batch:
            input_ids = el['input_ids']
            input_ids_padded = list_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask = el['attention_mask']
            attention_mask_padded = list_pad(attention_mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_mask_list)
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'binary_label': torch.tensor([el['binary_label'] for el in batch]),
        }

    def dag_collate(self, batch):
        assert len(batch) > 0
        max_len = max([len(el['encodings']['input_ids']) for el in batch])
        input_ids_list = []
        attention_mask_list = []
        step_indices_list = []
        step_indices_tokens_list = []
        head_indices_list = []
        head_indices_tokens_list = []
        batch_edge_list_words = []
        batch_edge_list_tokens = []
        for el in batch:
            input_ids = el['encodings']['input_ids']
            input_ids_padded = list_pad(input_ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded))
            attention_mask_padded = list_pad(el['encodings']['attention_mask'], pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded))
            step_indices_padded = list_pad(el['step_indices_tokens'], pad_element=0, pad_length=max_len)
            step_indices_list.append(torch.tensor(step_indices_padded))
            step_indices_tokens_padded = list_pad(el['step_indices_tokens'], pad_element=0, pad_length=max_len)
            step_indices_tokens_list.append(torch.tensor(step_indices_tokens_padded))
            head_indices_padded = list_pad(el['head_indices_tokens'], pad_element=0, pad_length=max_len)
            head_indices_list.append(torch.tensor(head_indices_padded))
            head_indices_padded_tokens = list_pad(el['head_indices_tokens'], pad_element=0, pad_length=max_len)
            head_indices_tokens_list.append(torch.tensor(head_indices_padded_tokens))
            batch_edge_list_words.append(el['G_words'])
            batch_edge_list_tokens.append(el['G_tokens'])
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
            'G_words': batch_edge_list_words,
            'G_tokens': batch_edge_list_tokens,
        }

class ProcTextDataset(Dataset):
    def __init__(self,
                data: List[Dict],
                tokenizer,
                do_tokenize = True,
                do_add_bos = False,
                do_add_eos = False,
                disable_tqdm = False,
                ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.disable_tqdm = disable_tqdm
        if do_tokenize:
            self.tokenize()  
        self.make_edges()
        if do_add_bos:
            self.add_bos()
        if do_add_eos:
            self.add_eos()
    
    def filter_non_dags(self):
        filtered = []
        for i in tqdm(range(len(self.data)), desc = 'filter_non_dags', disable=self.disable_tqdm):
            if not nx.is_directed_acyclic_graph(self.data[i]['G_tokens']):
                continue
            else:
                filtered.append(self.data[i])
        self.data = filtered

    def filter_short_dags(self, k = 2):
        filtered = []
        for i in tqdm(range(len(self.data)), desc = 'filter_short_dags', disable=self.disable_tqdm):
            if nx.dag_longest_path_length(self.data[i]['G_tokens']) < k:
                continue
            else:
                filtered.append(self.data[i])
        self.data = filtered
    
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
        for i in tqdm(range(len(self.data)), desc = 'tokenize', disable=self.disable_tqdm):
            self.data[i]['encodings'] = self.tokenizer(self.data[i]['words'], is_split_into_words = True)
            word_ids = self.data[i]['encodings'].word_ids()
            self.data[i]['step_indices_tokens'] = [0 if word_idx is None else self.data[i]['step_indices'][word_idx] for word_idx in word_ids]
            self.data[i]['head_indices_tokens'] = self.get_head_indices_tokens(word_ids, self.data[i]['head_indices'])
    
    def add_eos(self):
        for i in tqdm(range(len(self.data)), desc = 'add_eos', disable=self.disable_tqdm):
            self.data[i]['encodings']['input_ids'] = self.data[i]['encodings']['input_ids'] + [self.tokenizer.eos_token_id]
            self.data[i]['encodings']['attention_mask'] = self.data[i]['encodings']['attention_mask'] + [1]
            self.data[i]['step_indices_tokens'] = self.data[i]['step_indices_tokens'] + [0]
            self.data[i]['head_indices_tokens'] = self.data[i]['head_indices_tokens'] + [0]

    def add_bos(self):
        for i in tqdm(range(len(self.data)), desc = 'add_bos', disable=self.disable_tqdm):
            self.data[i]['encodings']['input_ids'] = (
                [self.tokenizer.bos_token_id] +
                self.data[i]['encodings']['input_ids']
                )
            self.data[i]['encodings']['attention_mask'] = [1] + self.data[i]['encodings']['attention_mask']
            self.data[i]['step_indices_tokens'] = [0] + self.data[i]['step_indices_tokens']
            shifted_head_indices_tokens = [0 if el == 0 else el + 1 for el in self.data[i]['head_indices_tokens']]
            self.data[i]['head_indices_tokens'] = [0] + shifted_head_indices_tokens

    def make_edges(self):
        for i in tqdm(range(len(self.data)), desc = 'make_edges', disable=self.disable_tqdm):
            G_words = self.graph_from_erfgc(
                self.data[i]['head_indices'],
                self.data[i]['step_indices'],
                )
            self.data[i]['G_words'] = G_words
            G_tokens = self.graph_from_erfgc(
                self.data[i]['head_indices_tokens'],
                self.data[i]['step_indices_tokens'],
                )
            self.data[i]['G_tokens'] = G_tokens
            assert G_words.edges == G_tokens.edges, 'word- and token-wise edges are not identical'       

    def graph_from_erfgc(self, head_indices, step_indices):
        G = nx.DiGraph()
        edges = self.make_edge_list(head_indices=head_indices, step_indices=step_indices)
        G.add_edges_from(edges)
        unique_steps = set(step_indices) - {0}
        G.add_nodes_from(unique_steps)
        return G

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
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class Seq2SeqDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 max_length=1024,
                 prompt_type='minimal',
                 loss_type='full',
                 ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type
        if prompt_type == 'minimal':
            self.make_pair_samples_minimal()
        elif prompt_type == 'natlang':
            self.make_pair_samples_natlang()
        elif prompt_type == 'only_shuffled':
            self.make_one_sided_samples(side = 'shuf')
        elif prompt_type == 'only_original':
            self.make_one_sided_samples(side = 'orig')
        self.prune_longs()

    def make_pair_samples_minimal(self, batch_size=1000):
        total_len = len(self.data)
        
        for start_idx in tqdm(range(0, total_len, batch_size), desc='tokenizing...'):
            end_idx = min(start_idx + batch_size, total_len)
            batch_slice = self.data[start_idx:end_idx]
            
            shuf_batch = [' '.join(item['shuf']) + ' ' for item in batch_slice]
            orig_batch = [' '.join(item['orig']) for item in batch_slice]
            enc_shuf = self.tokenizer(shuf_batch, add_special_tokens=False)
            enc_orig = self.tokenizer(orig_batch, add_special_tokens=False)
            
            for i, (shuf_ids, shuf_mask, orig_ids, orig_mask) in enumerate(zip(
                enc_shuf['input_ids'], 
                enc_shuf['attention_mask'], 
                enc_orig['input_ids'], 
                enc_orig['attention_mask']
            )):
                current_input_ids = shuf_ids + orig_ids + [self.tokenizer.eos_token_id]
                if self.loss_type == 'prompt_only_loss':
                    current_attention_mask = [0] * len(shuf_mask) + orig_mask + [1]
                elif self.loss_type == 'full_loss':
                    current_attention_mask = shuf_mask + orig_mask + [1]

                # Assignment only occurs for valid lengths
                self.data[start_idx + i]['text'] = shuf_batch[i] + orig_batch[i]
                self.data[start_idx + i]['input_ids'] = current_input_ids
                self.data[start_idx + i]['attention_mask'] = current_attention_mask

    def format_recipe_sample(self, batch_steps_scrambled, batch_steps_ordered):
        scrambled_text = "\n- ".join(batch_steps_scrambled)
        ordered_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(batch_steps_ordered)]
        )
        prompt = (
            f"Below is a jumbled list of recipe steps. Put them in the correct order.\n\n"
            f"Input:\n- {scrambled_text}\n\n"
            f"Correct order:\n"
        )
        completion = (
            f"{ordered_text}{self.tokenizer.eos_token}"
        )
        
        return prompt, completion

    def make_pair_samples_natlang(self, batch_size=1000):
        total_len = len(self.data)
        
        for start_idx in tqdm(range(0, total_len, batch_size), desc='making samples...'):
            end_idx = min(start_idx + batch_size, total_len)
            batch_slice = self.data[start_idx:end_idx]
            
            for i, item in enumerate(batch_slice):
                training_sample = self.format_recipe_sample(item['shuf'], item['orig'])
                encoding = self.tokenizer(training_sample, add_special_tokens=False)
                shuf_ids = encoding['input_ids'][0]
                shuf_mask = encoding['attention_mask'][0]
                orig_ids = encoding['input_ids'][1]
                orig_mask = encoding['attention_mask'][1]
                
                current_input_ids = shuf_ids + orig_ids
                if self.loss_type == 'prompt_only_loss':
                    current_attention_mask = [0] * len(shuf_mask) + orig_mask
                elif self.loss_type == 'full_loss':
                    current_attention_mask = shuf_mask + orig_mask

                self.data[start_idx + i]['text'] = training_sample[0] + training_sample[1]
                self.data[start_idx + i]['input_ids'] = current_input_ids
                self.data[start_idx + i]['attention_mask'] = current_attention_mask

    def make_one_sided_samples(self, side = 'orig', batch_size=1000):
        total_len = len(self.data)
        
        for start_idx in tqdm(range(0, total_len, batch_size), desc='tokenizing...'):
            end_idx = min(start_idx + batch_size, total_len)
            batch_slice = self.data[start_idx:end_idx]
            batch = [' '.join(item[side]) for item in batch_slice]            
            enc = self.tokenizer(batch, add_special_tokens=False)

            for i, (ids, mask) in enumerate(zip(enc['input_ids'], enc['attention_mask'])):
                self.data[start_idx + i]['text'] = batch[i]
                self.data[start_idx + i]['input_ids'] = ids + [self.tokenizer.eos_token_id]
                self.data[start_idx + i]['attention_mask'] = mask + [1]

    def prune_longs(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc = 'Pruning long samples...'):
            if len(self.data[i]['input_ids']) > self.max_length:
                continue
            else:
                data_pruned.append(self.data[i])
        print(f"Pruned {len(self.data) - len(data_pruned)} sequences exceeding {self.max_length} tokens.")
        self.data = data_pruned

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def read_csv_chunks(filepath, num_samples):
    if num_samples == 0:
        chunk_size = 100000
        chunks = []
        # Iterate over the file in chunks
        with pd.read_csv(filepath, chunksize=chunk_size) as reader:
            for chunk in reader:
                chunks.append(chunk)
        df = pd.concat(chunks, axis=0, ignore_index=True)
    else:
        # Load specific number of rows
        df = pd.read_csv(filepath, nrows=num_samples)
    return df