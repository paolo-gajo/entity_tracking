import torch
from torch.utils.data import Dataset
from typing import List, Dict
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
                 attn_mask_type='full',
                 loss_mask_type='completion_only',
                 batch_mode='random_samples',
                 min_recipe_steps=0,
                 max_recipe_steps=32,
                 step_token_id_map=None,
                 prepend_bos=False,
                 ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.attn_mask_type = attn_mask_type
        self.loss_mask_type = loss_mask_type
        self.min_recipe_steps = min_recipe_steps
        self.max_recipe_steps = max_recipe_steps
        self.prompt_type = prompt_type
        self.batch_mode = batch_mode
        self.step_token_id_map = step_token_id_map  # {0: id_for_<step_0>, 1: id_for_<step_1>, ...}
        self.prepend_bos = prepend_bos

        if self.prompt_type == 'minimal_pairs':
            self.make_fn = self.make_minimal_pair_samples
        elif self.prompt_type == 'natlang_pairs':
            self.make_fn = self.make_natlang_pair_samples
        elif self.prompt_type == 'minimal_mono':
            self.make_fn = self.make_mono_samples
        elif self.prompt_type == 'step_token_pairs':
            self.make_fn = self.make_step_token_pair_samples
        elif self.prompt_type == 'pooled_pairs':
            self.make_fn = self.make_pooled_pair_samples

        if self.batch_mode == 'pos_neg':
            self.make_pos_neg_dataset()
        elif self.batch_mode == 'random_samples':
            self.make_random_samples_dataset()

        self.prune_long_token_lengths()
        self.prune_short_step_lengths()
        self.prune_long_step_lengths()

    def prune_long_token_lengths(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc='Pruning long token length samples...'):
            if isinstance(self.data[i], list):
                if all(len(self.data[i][j]['input_ids']) <= self.max_length for j in range(len(self.data[i]))):
                    data_pruned.append(self.data[i])
            else:
                if len(self.data[i]['input_ids']) <= self.max_length:
                    data_pruned.append(self.data[i])

        removed = len(self.data) - len(data_pruned)
        if removed > 0:
            print(f"Pruned {removed} sequences exceeding {self.max_length} tokens.")
        self.data = data_pruned

    def prune_short_step_lengths(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc='Pruning short step length samples...'):
            if isinstance(self.data[i], list):
                if max(self.data[i][0]['step_indices_mml']) >= self.min_recipe_steps:
                    data_pruned.append(self.data[i])
            else:
                if max(self.data[i]['step_indices_mml']) >= self.min_recipe_steps:
                    data_pruned.append(self.data[i])

        removed = len(self.data) - len(data_pruned)
        if removed > 0:
            print(f"Pruned {removed} sequences with fewer than {self.min_recipe_steps} steps.")
        self.data = data_pruned

    def prune_long_step_lengths(self):
        data_pruned = []
        for i in tqdm(range(len(self.data)), desc='Pruning long step length samples...'):
            if isinstance(self.data[i], list):
                if max(self.data[i][0]['step_indices_mml']) <= self.max_recipe_steps:
                    data_pruned.append(self.data[i])
            else:
                if max(self.data[i]['step_indices_mml']) <= self.max_recipe_steps:
                    data_pruned.append(self.data[i])

        removed = len(self.data) - len(data_pruned)
        if removed > 0:
            print(f"Pruned {removed} sequences exceeding {self.max_recipe_steps} steps.")
        self.data = data_pruned

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def make_random_samples_dataset(self):
        formatted_data = []
        chunk_size = 2048
        tbar = tqdm(range(0, len(self.data), chunk_size),
                    desc=f'Formatting random-samples dataset (prompt_type = {self.prompt_type})...')
        for i in tbar:
            chunk = self.data[i: i + chunk_size]
            formatted_data.extend(self.make_fn(chunk))
        self.data = formatted_data

    def make_pos_neg_dataset(self):
        formatted_data = []
        for batch in tqdm(self.data,
                          desc=f'Formatting pos-neg dataset (prompt_type = {self.prompt_type})...'):
            formatted_batch = self.make_fn(batch)
            formatted_data.append(formatted_batch)
        self.data = formatted_data

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _concat(chunks: List[List[int]]) -> List[int]:
        out: List[int] = []
        for ids in chunks:
            out.extend(ids)
        return out

    # ==================================================================
    #  EXISTING FORMATTERS  (unchanged — included for completeness)
    # ==================================================================

    def make_mono_samples(self, batch):
        formatted_batch = []
        for i, item in enumerate(batch):
            target_list = item['orig' if item['binary_label'] else 'shuf']
            input_ids = []
            step_indices = []
            for step_num, step_str in enumerate(target_list):
                if step_num > 0:
                    step_str = ' ' + step_str
                step_tokens = self.tokenizer.encode(step_str, add_special_tokens=False)
                input_ids.extend(step_tokens)
                step_indices.extend([step_num + 1] * len(step_tokens))
            input_ids.append(self.tokenizer.eos_token_id)
            step_indices.append(0)
            attention_mask = [1] * len(input_ids)
            formatted_batch.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'step_indices': step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    def make_minimal_pair_samples(self, batch):
        formatted_batch = []
        flat_src, flat_tgt = [], []
        for item in batch:
            flat_src.extend([" " + s.strip() for s in item['shuf']])
            flat_tgt.extend([" " + s.strip() for s in item['orig']])
        shuf_encs = self.tokenizer(flat_src, add_special_tokens=False)['input_ids']
        orig_encs = self.tokenizer(flat_tgt, add_special_tokens=False)['input_ids']
        sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
        shuf_idx, orig_idx = 0, 0
        for i, item in enumerate(batch):
            n_src = len(item['shuf'])
            n_tgt = len(item['orig'])
            src_chunks = shuf_encs[shuf_idx: shuf_idx + n_src]
            tgt_chunks = orig_encs[orig_idx: orig_idx + n_tgt]
            shuf_idx += n_src
            orig_idx += n_tgt

            src_ids = self._concat(src_chunks) + sep_ids
            src_step_indices = []
            for j, shuf_step in enumerate(item['shuf']):
                orig_idx_map = item['orig'].index(shuf_step) + 1
                chunk_len = len(src_chunks[j])
                src_step_indices.extend([orig_idx_map] * chunk_len)
            src_step_indices.extend([0] * len(sep_ids))

            tgt_ids = self._concat(tgt_chunks)
            tgt_step_indices = []
            for step_num, chunk in enumerate(tgt_chunks):
                tgt_step_indices.extend([step_num + 1] * len(chunk))

            bos_prefix = [self.tokenizer.bos_token_id] if self.prepend_bos else []
            n_bos = len(bos_prefix)
            input_ids = bos_prefix + src_ids + tgt_ids + [self.tokenizer.eos_token_id]
            step_indices_mml = [0] * n_bos + src_step_indices + [0] * len(tgt_step_indices) + [0]

            if 'completion_only' in self.attn_mask_type:
                attn_mask = [0] * (n_bos + len(src_ids)) + [1] * len(tgt_ids) + [1]
            elif 'full' in self.attn_mask_type:
                attn_mask = [1] * len(input_ids)
            else:
                raise ValueError(f"Unknown attn_mask_type: {self.attn_mask_type}")
            if 'completion_only' in self.loss_mask_type:
                loss_mask = [0] * (n_bos + len(src_ids)) + [1] * len(tgt_ids) + [1]
            elif 'full' in self.loss_mask_type:
                loss_mask = [1] * len(input_ids)
            else:
                raise ValueError(f"Unknown loss_mask_type: {self.loss_mask_type}")
            formatted_batch.append({
                'input_ids': input_ids,
                'attn_mask': attn_mask,
                'loss_mask': loss_mask,
                'step_indices_mml': step_indices_mml,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    def make_natlang_pair_samples(self, batch):
        formatted_batch = []
        for i, item in enumerate(batch):
            scrambled_text = "\n- ".join(item['shuf'])
            prompt_str = (
                f"Below is a jumbled list of recipe steps. Put them in the correct order.\n\n"
                f"Input:\n- {scrambled_text}\n\n"
                f"Correct order:\n"
            )
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_step_indices = [0] * len(prompt_ids)
            target_ids, target_step_indices = [], []
            for step_num, step_str in enumerate(item['orig']):
                prefix = "" if step_num == 0 else "\n"
                current_step_text = f"{prefix}{step_num + 1}. {step_str}"
                step_tokens = self.tokenizer.encode(current_step_text, add_special_tokens=False)
                target_ids.extend(step_tokens)
                target_step_indices.extend([step_num + 1] * len(step_tokens))
            input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
            step_indices = prompt_step_indices + target_step_indices + [0]
            if 'completion_only' in self.attn_mask_type:
                attention_mask = [0] * len(prompt_ids) + [1] * len(target_ids) + [1]
            elif 'full' in self.attn_mask_type:
                attention_mask = [1] * len(input_ids)
            else:
                raise ValueError(f"Unknown attn_mask_type: {self.attn_mask_type}")
            formatted_batch.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'step_indices': step_indices,
                'binary_label': batch[i]['binary_label'],
            })
        return formatted_batch

    # ==================================================================
    #  NEW: Pooled-CLM pairs  (Section 3.1)
    #
    #  Same layout as minimal_pairs but the completion side also carries
    #  step indices so PooledCausalLMLoss can group tokens by step.
    # ==================================================================

    def make_pooled_pair_samples(self, batch):
        """
        Format:  π_shuf ⊕ sep ⊕ π_orig ⊕ eos

        Identical to make_minimal_pair_samples except that step_indices_mml
        is populated on BOTH sides:
          - prefix tokens get the *original* step index of the shuffled step
            they belong to  (same as before — needed for MML)
          - completion tokens get their sequential step index 1..N
            (needed for PooledCausalLMLoss)

        A new field `completion_step_indices` carries the completion-side
        indices in a separate tensor to keep the MML and pooled-CLM index
        semantics cleanly separated.
        """
        formatted_batch = []

        flat_src, flat_tgt = [], []
        for item in batch:
            flat_src.extend([" " + s.strip() for s in item['shuf']])
            flat_tgt.extend([" " + s.strip() for s in item['orig']])

        shuf_encs = self.tokenizer(flat_src, add_special_tokens=False)['input_ids']
        orig_encs = self.tokenizer(flat_tgt, add_special_tokens=False)['input_ids']
        sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)

        shuf_idx, orig_idx = 0, 0
        for i, item in enumerate(batch):
            n_src = len(item['shuf'])
            n_tgt = len(item['orig'])
            src_chunks = shuf_encs[shuf_idx: shuf_idx + n_src]
            tgt_chunks = orig_encs[orig_idx: orig_idx + n_tgt]
            shuf_idx += n_src
            orig_idx += n_tgt

            # --- prefix (shuffled) ------------------------------------------
            src_ids = self._concat(src_chunks) + sep_ids

            # MML step indices on the prefix: map each shuffled step back to
            # its original position
            src_step_indices_mml = []
            for j, shuf_step in enumerate(item['shuf']):
                orig_pos = item['orig'].index(shuf_step) + 1
                src_step_indices_mml.extend([orig_pos] * len(src_chunks[j]))
            src_step_indices_mml.extend([0] * len(sep_ids))

            # --- completion (original order) --------------------------------
            tgt_ids = self._concat(tgt_chunks)

            tgt_step_indices = []
            for step_num, chunk in enumerate(tgt_chunks):
                tgt_step_indices.extend([step_num + 1] * len(chunk))

            # --- full sequence -----------------------------------------------
            input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]

            # MML indices: prefix has orig-position mapping; completion = 0
            step_indices_mml = src_step_indices_mml + [0] * len(tgt_ids) + [0]

            # Completion step indices for PooledCausalLMLoss:
            # prefix = 0, completion = 1..N, eos = 0
            completion_step_indices = (
                [0] * len(src_ids) + tgt_step_indices + [0]
            )

            # --- masks --------------------------------------------------------
            if 'completion_only' in self.attn_mask_type:
                attn_mask = [0] * len(src_ids) + [1] * len(tgt_ids) + [1]
            elif 'full' in self.attn_mask_type:
                attn_mask = [1] * len(input_ids)
            else:
                raise ValueError(f"Unknown attn_mask_type: {self.attn_mask_type}")

            if 'completion_only' in self.loss_mask_type:
                loss_mask = [0] * len(src_ids) + [1] * len(tgt_ids) + [1]
            elif 'full' in self.loss_mask_type:
                loss_mask = [1] * len(input_ids)
            else:
                raise ValueError(f"Unknown loss_mask_type: {self.loss_mask_type}")

            formatted_batch.append({
                'input_ids': input_ids,
                'attn_mask': attn_mask,
                'loss_mask': loss_mask,
                'step_indices_mml': step_indices_mml,
                'completion_step_indices': completion_step_indices,
                'binary_label': item['binary_label'],
            })
        return formatted_batch

    # ==================================================================
    #  Step Token Prediction pairs  (Section 3.2)
    #
    #  Format:
    #    prefix  =  <step_σ(1)> S_σ(1) <step_σ(2)> S_σ(2) … sep
    #    completion =  <step_1> <step_2> … <step_N> eos
    #
    #  Step tokens are real tokens in the vocabulary.  The model predicts
    #  them via its standard LM head (no separate classifier).
    # ==================================================================

    def make_step_token_pair_samples(self, batch):
        """
        Constructs sequences with real step-token vocabulary entries.

        Each step in the shuffled prefix is appended with its step token.
        The completion consists of step tokens interleaved with natural 
        text in the correct topological order.

        Requires ``self.step_token_id_map`` to be set — a dict mapping
        0-indexed step index → token id in the tokenizer.
        """
        assert self.step_token_id_map is not None, (
            "step_token_pairs requires step_token_id_map to be passed to Seq2SeqDataset"
        )
        formatted_batch = []

        # Batch-tokenize all step strings
        flat_steps = []
        for item in batch:
            flat_steps.extend([" " + s.strip() for s in item['shuf']])
        step_encs = self.tokenizer(flat_steps, add_special_tokens=False)['input_ids']

        sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)

        enc_idx = 0
        for i, item in enumerate(batch):
            n_steps = len(item['shuf'])
            chunks = step_encs[enc_idx: enc_idx + n_steps]
            enc_idx += n_steps

            # Skip recipes with more steps than available step tokens
            if n_steps > len(self.step_token_id_map):
                continue

            # ---- build prefix ------------------------------------------------
            prefix_ids = []
            prefix_step_indices_mml = []
            loss_mask_prefix = [] # For standard text tokens (CLM) in prefix

            for j, shuf_step in enumerate(item['shuf']):
                # 0-indexed original position (for MML step indices)
                orig_idx = item['orig'].index(shuf_step)

                # Regular content tokens
                prefix_ids.extend(chunks[j])
                prefix_step_indices_mml.extend([orig_idx + 1] * len(chunks[j]))
                loss_mask_prefix.extend([1] * len(chunks[j]))

                # Step token assigned by shuffled order (j-th step in prefix
                # gets <step_j>), appended after the step content
                
                # NOTE: Step j is associated the j-th element of the step_token_id_map.
                # This means that the step token is always found at a random step slot
                # and also at a random absolute token position in the input sequence.
                # This is very important because it prevents the model from cheating
                # and using positional information when predicting the step tokens
                # in the completion.
                prefix_ids.append(self.step_token_id_map[j])
                prefix_step_indices_mml.append(orig_idx + 1)  # 1-indexed for MML
                loss_mask_prefix.append(0) # Step tokens are excluded from CLM loss

            # Separator
            prefix_ids.extend(sep_ids)
            prefix_step_indices_mml.extend([0] * len(sep_ids))
            loss_mask_prefix.extend([1] * len(sep_ids))

            n_prefix = len(prefix_ids)

            # ---- build completion --------------------------------------------
            # The completion lists step tokens in the ORIGINAL order.
            # Since step token j was attached to shuf[j], we need to emit
            # them in the order that reconstructs orig.
            #
            # Example: shuf = [S3, S1, S2] → step tokens <step_0>=S3,
            #          <step_1>=S1, <step_2>=S2
            #          orig order is S1, S2, S3 → completion = <step_1> [Text 1] <step_2> [Text 2] <step_0> [Text 3]
            comp_ids = []
            comp_step_indices_mml = []
            loss_mask_completion = [] # For standard text tokens (CLM)
            stp_mask_completion = []  # For step tokens (STP)

            for orig_pos in range(n_steps):
                # Which shuffled index holds orig step orig_pos?
                orig_step = item['orig'][orig_pos]
                shuf_j = item['shuf'].index(orig_step)
                
                # 1. Emit Step token (Planning what goes next)
                stp_id = self.step_token_id_map[shuf_j]
                comp_ids.append(stp_id)
                comp_step_indices_mml.append(orig_pos + 1)  # 1-indexed
                loss_mask_completion.append(0) # Not part of standard CLM
                stp_mask_completion.append(1)  # Included in STP
                
                # 2. Emit Step text (Grounding it into natural language)
                text_chunk = chunks[shuf_j]
                comp_ids.extend(text_chunk)
                comp_step_indices_mml.extend([orig_pos + 1] * len(text_chunk))
                loss_mask_completion.extend([1] * len(text_chunk))
                stp_mask_completion.extend([0] * len(text_chunk))

            # ---- assemble full sequence + EOS --------------------------------
            bos_prefix = [self.tokenizer.bos_token_id] if self.prepend_bos else []
            input_ids = bos_prefix + prefix_ids + comp_ids + [self.tokenizer.eos_token_id]
            attn_mask = [1] * len(input_ids)

            # MML step indices: prefix gets step ids, completion step tokens
            # also get their step ids (useful for geometry evaluation),
            # EOS gets 0.
            n_bos = len(bos_prefix)
            step_indices_mml = (
                [0] * n_bos
                + prefix_step_indices_mml
                + comp_step_indices_mml
                + [0]
            )

            # CLM loss mask (vocab tokens + EOS)
            if 'completion_only' in self.loss_mask_type:
                loss_mask = [0] * (n_bos + n_prefix) + loss_mask_completion + [1]
                # The last separator token also gets 1 so the model learns to
                # predict the first step token across the boundary in CLM mode.
                loss_mask[n_bos + n_prefix - 1] = 1
            elif 'full' in self.loss_mask_type:
                loss_mask = [0] * n_bos + loss_mask_prefix + loss_mask_completion + [1]
            else:
                raise ValueError(f"Unknown loss_mask_type: {self.loss_mask_type}")

            # STP loss mask (step tokens only)
            stp_mask = [0] * (n_bos + n_prefix) + stp_mask_completion + [0]

            formatted_batch.append({
                'input_ids': input_ids,
                'attn_mask': attn_mask,
                'loss_mask': loss_mask,
                'stp_mask': stp_mask,
                'step_indices_mml': step_indices_mml,
                'binary_label': item['binary_label'],
            })

        return formatted_batch


# ======================================================================
# Collator
# ======================================================================

def list_pad(t, pad_element, pad_length=0):
    return t + [pad_element] * max(0, (pad_length - len(t)))


def tensor_pad(t, pad_element, pad_length=0, side='right'):
    def apply_padding(t, t_pad, dim, side=side):
        if side == 'right':
            return torch.cat([t, t_pad], dim=dim)
        else:
            return torch.cat([t_pad, t], dim=dim)
    assert t.dim() > 0
    if t.dim() == 1:
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(0)), dtype=t.dtype)
        return apply_padding(t, t_pad, dim=0)
    elif t.dim() == 2:
        B, _ = t.shape
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(1)), dtype=t.dtype)
        t_pad = t_pad.expand(B, pad_length)
        return apply_padding(t, t_pad, dim=1)
    elif t.dim() == 3:
        B, _, D = t.shape
        t_pad = torch.tensor([pad_element] * (pad_length - t.size(1)), dtype=t.dtype)
        t_pad = t_pad.expand(B, pad_length, D)
        return apply_padding(t, t_pad, dim=1)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def seq2seq_collate(self, batch):
        if isinstance(batch[0], list):
            flat_batch = []
            for group in batch:
                flat_batch.extend(group)
            batch = flat_batch

        def get_len(seq):
            return seq.size(0) if isinstance(seq, torch.Tensor) else len(seq)

        max_len = max([get_len(el['input_ids']) for el in batch])

        input_ids_list = []
        attn_mask_list = []
        loss_mask_list = []
        step_indices_mml_list = []
        completion_step_indices_list = []
        binary_labels_list = []

        # Step-token prediction fields
        step_token_ids_list = []
        step_token_mask_list = []
        stp_labels_list = []
        has_stp_mask = 'stp_mask' in batch[0]

        stp_mask_list = []

        has_step_token = 'step_token_ids' in batch[0]
        has_stp_labels = 'stp_labels' in batch[0]
        has_completion_step_indices = 'completion_step_indices' in batch[0]

        for el in batch:
            ids = el['input_ids']
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            input_ids_list.append(
                torch.tensor(list_pad(ids, self.tokenizer.pad_token_id, max_len), dtype=torch.long))

            mask = el['attn_mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()
            attn_mask_list.append(
                torch.tensor(list_pad(mask, 0, max_len), dtype=torch.long))

            l_mask = el['loss_mask']
            if isinstance(l_mask, torch.Tensor):
                l_mask = l_mask.tolist()
            loss_mask_list.append(
                torch.tensor(list_pad(l_mask, 0, max_len), dtype=torch.long))

            if 'step_indices_mml' in el:
                steps_mml = el['step_indices_mml']
                if isinstance(steps_mml, torch.Tensor):
                    steps_mml = steps_mml.tolist()
                step_indices_mml_list.append(
                    torch.tensor(list_pad(steps_mml, 0, max_len), dtype=torch.long))

            if has_completion_step_indices:
                csi = el['completion_step_indices']
                if isinstance(csi, torch.Tensor):
                    csi = csi.tolist()
                completion_step_indices_list.append(
                    torch.tensor(list_pad(csi, 0, max_len), dtype=torch.long))

            if has_step_token:
                st_ids = el['step_token_ids']
                if isinstance(st_ids, torch.Tensor):
                    st_ids = st_ids.tolist()
                step_token_ids_list.append(
                    torch.tensor(list_pad(st_ids, 0, max_len), dtype=torch.long))

                st_mask = el['step_token_mask']
                if isinstance(st_mask, torch.Tensor):
                    st_mask = st_mask.tolist()
                step_token_mask_list.append(
                    torch.tensor(list_pad(st_mask, 0, max_len), dtype=torch.long))

            if has_stp_labels:
                sl = el['stp_labels']
                if isinstance(sl, torch.Tensor):
                    sl = sl.tolist()
                stp_labels_list.append(
                    torch.tensor(list_pad(sl, -100, max_len), dtype=torch.long))

            if has_stp_mask:
                sm = el['stp_mask']
                if isinstance(sm, torch.Tensor):
                    sm = sm.tolist()
                stp_mask_list.append(
                    torch.tensor(list_pad(sm, 0, max_len), dtype=torch.long))

            if 'binary_label' in el:
                binary_labels_list.append(el['binary_label'])

        batch_dict = {
            'input_ids': torch.stack(input_ids_list),
            'attn_mask': torch.stack(attn_mask_list),
            'loss_mask': torch.stack(loss_mask_list),
        }

        if step_indices_mml_list:
            batch_dict['step_indices_mml'] = torch.stack(step_indices_mml_list)

        if completion_step_indices_list:
            batch_dict['completion_step_indices'] = torch.stack(completion_step_indices_list)

        if step_token_ids_list:
            batch_dict['step_token_ids'] = torch.stack(step_token_ids_list)

        if step_token_mask_list:
            batch_dict['step_token_mask'] = torch.stack(step_token_mask_list)

        if stp_labels_list:
            batch_dict['stp_labels'] = torch.stack(stp_labels_list)

        if stp_mask_list:
            batch_dict['stp_mask'] = torch.stack(stp_mask_list)

        if binary_labels_list:
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
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'step_indices': torch.stack(step_indices_list),
            'head_indices': torch.stack(head_indices_list),
            'step_indices_tokens': torch.stack(step_indices_tokens_list),
            'head_indices_tokens': torch.stack(head_indices_tokens_list),
            'G_words': batch_edge_list_words,
            'G_tokens': batch_edge_list_tokens,
        }


# ======================================================================
# Dataset construction helpers  (unchanged)
# ======================================================================

def make_pos_neg_samples_dataset(data, k=1):
    step_list_orig = [x['directions'] for x in data if len(set(x['directions'])) > 1]
    print(f"Filtered Dataset Size: {len(step_list_orig)}")
    dataset = []
    for orig in tqdm(step_list_orig, desc='Generating batches...'):
        n_steps = len(orig)
        batch = []
        batch.append({'orig': orig, 'shuf': orig, 'binary_label': 1})
        max_possible_negatives = math.factorial(n_steps) - 1
        target_negatives = k - 1
        num_negatives = min(target_negatives, max_possible_negatives)
        selected_shuffles = []
        if n_steps <= 8:
            all_perms = [list(p) for p in permutations(orig) if list(p) != orig]
            selected_shuffles = random.sample(all_perms, num_negatives)
        else:
            seen_hashes = set()
            while len(selected_shuffles) < num_negatives:
                shuf = random.sample(orig, n_steps)
                shuf_tuple = tuple(shuf)
                if shuf_tuple not in seen_hashes and shuf != orig:
                    seen_hashes.add(shuf_tuple)
                    selected_shuffles.append(shuf)
        for shuf in selected_shuffles:
            batch.append({'orig': orig, 'shuf': shuf, 'binary_label': 0})
        dataset.append(batch)
    return dataset


def make_random_samples_dataset(data, neg_ratio=0.5):
    print(f"Dataset Size: {len(data)}")
    step_list_orig = [x['directions'] for x in tqdm(data, desc='Filtering...') if len(set(x['directions'])) > 1]
    total_samples = len(step_list_orig)
    halfway_point = int(total_samples * neg_ratio)
    sample = random.sample
    data_pairs = []
    for i in tqdm(range(halfway_point), desc='Sampling negatives...'):
        orig = step_list_orig[i]
        n_steps = len(orig)
        shuf = sample(orig, n_steps)
        while shuf == orig:
            shuf = sample(orig, n_steps)
        data_pairs.append({'orig': orig, 'shuf': shuf, 'binary_label': int(orig == shuf)})
    for i in tqdm(range(halfway_point, total_samples), desc='Copying ground-truths...'):
        orig = step_list_orig[i]
        shuf = orig
        data_pairs.append({'orig': orig, 'shuf': shuf, 'binary_label': int(orig == shuf)})
    binary_labels_positive = sum(d['binary_label'] for d in data_pairs)
    pos_ratio = binary_labels_positive / len(data_pairs) if data_pairs else 0.0
    print(f'Positive/negative sample ratio: {pos_ratio:.6f}')
    return data_pairs


def pad_collate(batch, tokenizer, side='right'):
    max_len = max(len(el['input_ids']) for el in batch)
    input_ids = []
    attention_masks = []
    labels = []
    for el in batch:
        ids = tensor_pad(el['input_ids'], tokenizer.pad_token_id, max_len, side=side)
        mask = tensor_pad(el['attention_mask'], 0, max_len, side=side)
        input_ids.append(torch.tensor(ids))
        attention_masks.append(torch.tensor(mask))
        if 'label' in el:
            labels.append(el['label'])
    batch_dict = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
    }
    if labels:
        batch_dict['label'] = torch.tensor(labels, dtype=torch.long)
    return batch_dict


def prepare_text_batch_prompt(batch, tokenizer):
    prompt = ''
    for i in range(batch['input_ids'].shape[0]):
        decoded_all = tokenizer.decode(batch['input_ids'][i])
        clm_mask = batch['loss_mask'][i] == 1
        valid_inputs = batch['input_ids'][i][clm_mask]
        decoded_valid = tokenizer.decode(valid_inputs)
        stp_decoded = ''
        if 'stp_mask' in batch:
            stp_mask = batch['stp_mask'][i] == 1
            valid_inputs_stp = batch['input_ids'][i][stp_mask]
            stp_decoded = tokenizer.decode(valid_inputs_stp)
        prompt += (decoded_all +
                   '\n\n' +
                   ('-' * 100) +
                   '\n\n' +
                   decoded_valid +
                   '\n\n' +
                   f"Binary label: {batch['binary_label'][i]}" +
                   '\n\n' +
                   ('#' * 100) +
                   '\n\n' +
                   f'step_decoded: {stp_decoded}' +
                   '\n\n' +
                   ('=' * 100) +
                   '\n\n'
                   )
    return prompt


# ======================================================================
# Other Dataset classes  (ICLDataset, ProcTextDataset — unchanged)
# ======================================================================

class ICLDataset(Dataset):
    def __init__(self, icl_dataset, test_dataset, tokenizer, n_icl,
                 max_length=1024, prune_lengths=True, num_samples=0):
        super().__init__()
        self.icl_dataset = icl_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.n_icl = n_icl
        self.max_length = max_length
        self.data = self.test_dataset.apply(lambda x: self.make_icl_sample(x), axis=1)
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

    def format_steps(self, line, append_labels=False, labels_nl=('no', 'yes')):
        steps = line['steps']
        steps_joined = 'Steps: ' + ' '.join(steps)
        idx_head = line['step_pair_idx_asked_about'][0]
        head = f" Step {line['step_pair_idx_asked_about'][0] + 1}: " + steps[idx_head]
        idx_tail = line['step_pair_idx_asked_about'][1]
        tail = f" Step {line['step_pair_idx_asked_about'][1] + 1}: " + steps[idx_tail]
        question = f" {line['binary_question']}"
        label_text = labels_nl[line['label']]
        answer = f" Answer: {label_text}"
        prompt = steps_joined + '\n\n' + head + '\n\n' + tail + '\n\n' + question + '\n\n'
        if append_labels:
            prompt = prompt + answer + '\n\n'
        prompt_tokens = self.tokenizer(prompt)['input_ids']
        return prompt_tokens

    def make_icl_sample(self, line, sample_type='real'):
        icl_dataset = self.icl_dataset
        if sample_type:
            icl_dataset = icl_dataset[icl_dataset['type'] == sample_type]
        df_icl = icl_dataset.groupby(
            ['label', 'type', 'direction'], group_keys=False
        ).sample(n=self.n_icl, replace=False)
        df_icl = df_icl.sample(frac=1)
        icl_input_ids = df_icl.apply(lambda x: self.format_steps(x, append_labels=True), axis=1)
        icl_input_ids_tensors = [torch.tensor(el) for el in icl_input_ids.to_list()]
        test_input_ids = self.format_steps(line, append_labels=False)
        test_input_ids_tensor = torch.tensor(test_input_ids)
        input_ids = torch.concat(icl_input_ids_tensors + [test_input_ids_tensor])
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids, dtype=input_ids.dtype),
            'label': int(line['label']),
        }


class ProcTextDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer,
                 do_tokenize=True, do_add_bos=False, do_add_eos=False,
                 disable_tqdm=False):
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
        for i in tqdm(range(len(self.data)), desc='filter_non_dags', disable=self.disable_tqdm):
            if nx.is_directed_acyclic_graph(self.data[i]['G_tokens']):
                filtered.append(self.data[i])
        self.data = filtered

    def filter_short_dags(self, k=2):
        filtered = []
        for i in tqdm(range(len(self.data)), desc='filter_short_dags', disable=self.disable_tqdm):
            if nx.dag_longest_path_length(self.data[i]['G_tokens']) >= k:
                filtered.append(self.data[i])
        self.data = filtered

    def get_head_indices_tokens(self, word_ids, word_edges):
        T = len(word_ids)
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
        for i in tqdm(range(len(self.data)), desc='tokenize', disable=self.disable_tqdm):
            self.data[i]['encodings'] = self.tokenizer(self.data[i]['words'], is_split_into_words=True)
            word_ids = self.data[i]['encodings'].word_ids()
            self.data[i]['step_indices_tokens'] = [
                0 if word_idx is None else self.data[i]['step_indices'][word_idx] for word_idx in word_ids
            ]
            self.data[i]['head_indices_tokens'] = self.get_head_indices_tokens(
                word_ids, self.data[i]['head_indices'])

    def add_eos(self):
        for i in tqdm(range(len(self.data)), desc='add_eos', disable=self.disable_tqdm):
            self.data[i]['encodings']['input_ids'] += [self.tokenizer.eos_token_id]
            self.data[i]['encodings']['attention_mask'] += [1]
            self.data[i]['step_indices_tokens'] += [0]
            self.data[i]['head_indices_tokens'] += [0]

    def add_bos(self):
        for i in tqdm(range(len(self.data)), desc='add_bos', disable=self.disable_tqdm):
            self.data[i]['encodings']['input_ids'] = (
                [self.tokenizer.bos_token_id] + self.data[i]['encodings']['input_ids']
            )
            self.data[i]['encodings']['attention_mask'] = [1] + self.data[i]['encodings']['attention_mask']
            self.data[i]['step_indices_tokens'] = [0] + self.data[i]['step_indices_tokens']
            shifted = [0 if el == 0 else el + 1 for el in self.data[i]['head_indices_tokens']]
            self.data[i]['head_indices_tokens'] = [0] + shifted

    def make_edges(self):
        for i in tqdm(range(len(self.data)), desc='make_edges', disable=self.disable_tqdm):
            G_words = self.graph_from_erfgc(self.data[i]['head_indices'], self.data[i]['step_indices'])
            self.data[i]['G_words'] = G_words
            G_tokens = self.graph_from_erfgc(self.data[i]['head_indices_tokens'], self.data[i]['step_indices_tokens'])
            self.data[i]['G_tokens'] = G_tokens
            assert G_words.edges == G_tokens.edges

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
