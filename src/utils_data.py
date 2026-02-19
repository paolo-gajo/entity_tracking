import torch
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm.auto import tqdm
from tqdm.auto import tqdm
import networkx as nx
import random
import math
from itertools import permutations
from tqdm import tqdm

class Seq2SeqDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 max_length=1024,
                 prompt_type='minimal',
                 attention_mask_type='full',
                 batch_mode = 'random_samples',
                 min_recipe_steps = 0,
                 ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.attention_mask_type = attention_mask_type
        self.min_recipe_steps = min_recipe_steps
        self.prompt_type = prompt_type
        self.batch_mode = batch_mode
        
        if self.prompt_type == 'minimal_pairs':
            self.make_fn = self.make_minimal_pair_samples
        elif self.prompt_type == 'natlang_pairs':
            self.make_fn = self.make_natlang_pair_samples
        elif self.prompt_type == 'only_shuffled':
            self.make_fn = lambda x: self.make_one_sided_samples(x, side='shuf')
        elif self.prompt_type == 'only_original':
            self.make_fn = lambda x: self.make_one_sided_samples(x, side='orig')
        elif self.prompt_type == 'minimal_mono':
            self.make_fn = self.make_mono_samples
        
        if self.batch_mode == 'pos_neg':
            self.make_pos_neg_dataset()
        elif self.batch_mode == 'random_samples':
            self.make_random_samples_dataset()
        
        self.prune_longs()
        self.prune_shorts()

    def prune_longs(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc='Pruning long samples...'):
            if isinstance(self.data[i], list):
                # Batched case: check the first element's input_ids length as representative
                if all(len(self.data[i][j]['input_ids']) <= self.max_length for j in range(len(self.data[i]))):
                    data_pruned.append(self.data[i])
            else:
                if len(self.data[i]['input_ids']) <= self.max_length:
                    data_pruned.append(self.data[i])
        
        removed = len(self.data) - len(data_pruned)
        if removed > 0:
            print(f"Pruned {removed} sequences exceeding {self.max_length} tokens.")
        self.data = data_pruned

    # prune short recipes
    def prune_shorts(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc='Pruning short samples...'):
            if isinstance(self.data[i], list):
                if max(self.data[i][0]['step_indices']) >= self.min_recipe_steps:
                    data_pruned.append(self.data[i])
            else:
                if max(self.data[i]['step_indices']) >= self.min_recipe_steps:
                    data_pruned.append(self.data[i])
        
        removed = len(self.data) - len(data_pruned)
        if removed > 0:
            print(f"Pruned {removed} sequences with fewer than {self.min_recipe_steps} steps.")
        self.data = data_pruned

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def make_random_samples_dataset(self):
        formatted_data = []
        for item in tqdm(self.data, desc=f'Formatting random-samples dataset (prompt_type = {self.prompt_type})...'):
            formatted_data.append(self.make_fn([item])[0])
        self.data = formatted_data

    def make_pos_neg_dataset(self):
        formatted_data = []
        for batch in tqdm(self.data, desc=f'Formatting pos-neg dataset (prompt_type = {self.prompt_type})...'):
            formatted_batch = self.make_fn(batch)
            formatted_data.append(formatted_batch)
        self.data = formatted_data

    def make_mono_samples(self, batch):
        formatted_batch = []
        for i, item in enumerate(batch):
            target_list = item['orig' if item['binary_label'] else 'shuf']
            
            full_input_ids = []
            full_step_indices = []
            
            for step_num, step_str in enumerate(target_list):
                if step_num > 0: step_str = ' ' + step_str
                
                # step_tokens = self.tokenizer.encode(step_str, add_special_tokens=False)
                step_tokens = self._tok_step(step_str)

                full_input_ids.extend(step_tokens)
                # Assign Index 1..N based on generation order
                full_step_indices.extend([step_num + 1] * len(step_tokens))
            
            # Add EOS
            full_input_ids.append(self.tokenizer.eos_token_id)
            full_step_indices.append(0) # EOS is 0
            
            # Mask (One-sided is always full loss usually, but consistent with class logic)
            full_attention_mask = [1] * len(full_input_ids)

            formatted_batch.append({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_mask,
                'step_indices': full_step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    def _tok_step(self, step: str) -> List[int]:
        # return self.tokenizer.encode(step.strip(), add_special_tokens=False)
        return self.tokenizer.encode(" " + step.strip(), add_special_tokens=False)

    def _tok_steps(self, steps: List[str]) -> List[List[int]]:
        return [self._tok_step(s) for s in steps]

    @staticmethod
    def _concat(chunks: List[List[int]]) -> List[int]:
        out: List[int] = []
        for ids in chunks:
            out.extend(ids)
        return out

    def make_minimal_pair_samples(self, batch):
        formatted_batch = []
        for i, item in enumerate(batch):
            shuf_chunks = self._tok_steps(item['shuf'])
            orig_chunks = self._tok_steps(item['orig'])

            # Prompt = concatenation of shuffled step chunks
            sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
            prompt_ids = self._concat(shuf_chunks) + sep_ids
            prompt_step_indices = [0] * len(prompt_ids)

            # Target = concatenation of original step chunks
            target_ids = self._concat(orig_chunks)

            # Step indices align with the ORIGINAL order (1..N over orig)
            target_step_indices = []
            for step_num, chunk in enumerate(orig_chunks):
                target_step_indices.extend([step_num + 1] * len(chunk))

            # Combine (+ EOS)
            full_input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            full_step_indices = prompt_step_indices + target_step_indices + [0]

            # Mask
            if 'completion_only' in self.attention_mask_type:
                full_attention_mask = [0] * len(prompt_ids) + [1] * len(target_ids) + [1]
            elif 'full' in self.attention_mask_type:
                full_attention_mask = [1] * len(full_input_ids)
            else:
                raise ValueError(f"Unknown attention_mask_type: {self.attention_mask_type}")

            formatted_batch.append({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_mask,
                'step_indices': full_step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    def make_natlang_pair_samples(self, batch):    
        formatted_batch = []
        for i, item in enumerate(batch):
            # 1. Construct Prompt (Index 0)
            scrambled_text = "\n- ".join(item['shuf'])
            prompt_str = (
                f"Below is a jumbled list of recipe steps. Put them in the correct order.\n\n"
                f"Input:\n- {scrambled_text}\n\n"
                f"Correct order:\n"
            )
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_step_indices = [0] * len(prompt_ids)
            
            # 2. Construct Target Step-by-Step (Index 1..N)
            target_ids = []
            target_step_indices = []
            
            for step_num, step_str in enumerate(item['orig']):
                # Format: "1. Step text"
                # We add a newline BEFORE steps 2..N to separate them
                prefix = ""
                if step_num > 0:
                    prefix = "\n"
                
                current_step_text = f"{prefix}{step_num + 1}. {step_str}"
                step_tokens = self.tokenizer.encode(current_step_text, add_special_tokens=False)
                
                target_ids.extend(step_tokens)
                target_step_indices.extend([step_num + 1] * len(step_tokens))
            
            # 3. Combine
            full_input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            full_step_indices = prompt_step_indices + target_step_indices + [0]
            
            # 4. Mask
            if 'completion_only' in self.attention_mask_type:
                full_attention_mask = [0] * len(prompt_ids) + [1] * len(target_ids) + [1]
            elif 'full' in self.attention_mask_type:
                full_attention_mask = [1] * len(full_input_ids)
            else:
                raise ValueError(f"Unknown attention_mask_type: {self.attention_mask_type}")
            
            formatted_batch.append({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_mask,
                'step_indices': full_step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    def make_one_sided_samples(self, batch, side='orig'):
        formatted_batch = []
        for i, item in enumerate(batch):
            target_list = item[side]
            
            full_input_ids = []
            full_step_indices = []
            
            for step_num, step_str in enumerate(target_list):
                if step_num > 0: step_str = ' ' + step_str
                
                step_tokens = self.tokenizer.encode(step_str, add_special_tokens=False)
                full_input_ids.extend(step_tokens)
                # Assign Index 1..N based on generation order
                full_step_indices.extend([step_num + 1] * len(step_tokens))
            
            # Add EOS
            full_input_ids.append(self.tokenizer.eos_token_id)
            full_step_indices.append(0) # EOS is 0
            
            # Mask (One-sided is always full loss usually, but consistent with class logic)
            full_attention_mask = [1] * len(full_input_ids)

            formatted_batch.append({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_mask,
                'step_indices': full_step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

def make_pos_neg_samples_dataset(data, k=1):
    """
    Creates batches of size k containing:
    - 1 Positive pair (Original, Original, Label=1)
    - k-1 Negative pairs (Original, Shuffled, Label=0)
    
    Handles small-sequence constraints to avoid infinite loops.
    """

    # Filter for valid sequences with at least 2 steps (cannot shuffle length 0 or 1)
    step_list_orig = [x['directions'] for x in data if len(set(x['directions'])) > 1]
    
    print(f"Filtered Dataset Size: {len(step_list_orig)}")
    dataset = []
    
    for orig in tqdm(step_list_orig, desc='Generating batches...'):
        n_steps = len(orig)
        batch = []
        
        # --- 1. Add Positive Sample ---
        # The model needs to see the correct ordering with a label of 1
        batch.append({
            'orig': orig,
            'shuf': orig,
            'binary_label': 1
        })
        
        # --- 2. determine Negative Count ---
        # We need k-1 negatives, but we cannot ask for more unique shuffles than exist.
        # Max unique shuffles = n! - 1 (the original)
        max_possible_negatives = math.factorial(n_steps) - 1
        target_negatives = k - 1
        num_negatives = min(target_negatives, max_possible_negatives)
        
        selected_shuffles = []

        # --- 3. Generate Negatives (Hybrid Strategy) ---
        
        # Strategy A: Deterministic (for small N)
        # Prevents infinite loops when N is small (e.g., N=3 has only 5 wrong perms)
        if n_steps <= 8:
            # Generate all permutations, remove the original
            all_perms = [list(p) for p in permutations(orig) if list(p) != orig]
            # Sample without replacement
            selected_shuffles = random.sample(all_perms, num_negatives)
            
        # Strategy B: Rejection Sampling (for large N)
        # Faster for long recipes where collision probability is near zero
        else:
            seen_hashes = set()
            while len(selected_shuffles) < num_negatives:
                shuf = random.sample(orig, n_steps)
                shuf_tuple = tuple(shuf)
                
                # Check 1: Is it unique in our current batch?
                # Check 2: Is it accidentally the original?
                if shuf_tuple not in seen_hashes and shuf != orig:
                    seen_hashes.add(shuf_tuple)
                    selected_shuffles.append(shuf)

        # --- 4. Append Negatives to Batch ---
        for shuf in selected_shuffles:
            batch.append({
                'orig': orig,
                'shuf': shuf,
                'binary_label': 0
            })
        dataset.append(batch)
        
    return dataset

def make_random_samples_dataset(data):
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
        """
        Handles both:
        1. Standard Datasets: batch = [dict, dict, ...]
        2. Grouped/Batched Datasets: batch = [[dict, dict], [dict, dict], ...]
        """
        
        # --- 1. FLATTENING LOGIC ---
        # Check if the first element is a list. If so, we have a batch of batches.
        if isinstance(batch[0], list):
            # Flatten the list of lists into a single list of dicts
            flat_batch = []
            for group in batch:
                flat_batch.extend(group)
            batch = flat_batch
            
        # --- 2. Standard Padding Logic ---
        
        # Helper to safely get length of list or tensor
        def get_len(seq):
            return seq.size(0) if isinstance(seq, torch.Tensor) else len(seq)

        # Determine max length in this specific batch
        max_len = max([get_len(el['input_ids']) for el in batch])
        
        input_ids_list = []
        attention_mask_list = []
        step_indices_list = []
        binary_labels_list = []

        for el in batch:
            # -- Input IDs --
            # We use the global list_pad function you defined in your script
            # Ensure input is a list before padding
            ids = el['input_ids']
            if isinstance(ids, torch.Tensor): ids = ids.tolist()
                
            input_ids_padded = list_pad(ids, pad_element=self.tokenizer.pad_token_id, pad_length=max_len)
            input_ids_list.append(torch.tensor(input_ids_padded, dtype=torch.long))
            
            # -- Attention Mask --
            mask = el['attention_mask']
            if isinstance(mask, torch.Tensor): mask = mask.tolist()
                
            attention_mask_padded = list_pad(mask, pad_element=0, pad_length=max_len)
            attention_mask_list.append(torch.tensor(attention_mask_padded, dtype=torch.long))
            
            # -- Step Indices --
            if 'step_indices' in el:
                steps = el['step_indices']
                if isinstance(steps, torch.Tensor): steps = steps.tolist()
                    
                step_indices_padded = list_pad(steps, pad_element=0, pad_length=max_len)
                step_indices_list.append(torch.tensor(step_indices_padded, dtype=torch.long))

            # -- Binary Labels --
            if 'binary_label' in el:
                binary_labels_list.append(el['binary_label'])

        # --- 3. Stack and Return ---
        batch_dict = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
        }
        
        if step_indices_list:
            batch_dict['step_indices'] = torch.stack(step_indices_list)

        if binary_labels_list:
            # Float is usually better for labels if you plan to use BCEWithLogitsLoss later
            batch_dict['binary_label'] = torch.tensor(binary_labels_list, dtype=torch.float)

        return batch_dict

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

def prepare_text_batch_prompt(batch, tokenizer):
    prompt = ''
    for i in range(batch['input_ids'].shape[0]):
        decoded_all = tokenizer.decode(batch['input_ids'][i])
        mask = batch['attention_mask'][i] == 1
        valid_inputs = batch['input_ids'][i][mask]
        decoded_valid = tokenizer.decode(valid_inputs)
        prompt += (decoded_all + '\n\n' + ('-' * 100) + '\n\n' + decoded_valid + '\n\n' + f"Binary label: {batch['binary_label'][i]}" + '\n\n' + ('#' * 100) + '\n\n' )
    return prompt
