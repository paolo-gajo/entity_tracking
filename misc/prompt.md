how can i do this? keep in mind that, after training, the model should still be able to work as a standard LLM. i want it to learn knowledge about reasoning about causal dependencies, but it still has to be an LM after that that i can evaluate on causal reasoning benchmarks.

We have a dataset of procedural texts where every sample is a list of strings [step_1, step_2, ..., step_N], N \in {4, ..., 12}, step_i = {token_j}_j=0^|step_i|. We do not have any annotations, but we do know that these are procedural texts and so the annotations would be DAGs, where the sink would usually be towards the end of the list, realistically almost always the last. We cannot guarantee however that the leaves would be towards the start. We want a sequence model such as a causal attention Transformer (e.g. GPT2) to learn the dependencies between steps, i.e. the directed NxN adjacency matrix. Currently, our approach is making the model learn to produce its hidden states so that the poooled step embeddings abide by the order-embedding loss from `ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE`, Vendrov et al. 2016.

The unsupervised data from RecipeNLG (~2.2M samples) contains this information:

[
    {
        "title":"Oatmeal Breakfast Cookies",
        "directions":[
            "Blend together all wet ingredients.",
            "Mix the flour, soda, bran and salt.",
            "Mix into wet ingredients.",
            "Blend in oatmeal.",
            "Drop by large tablespoons.",
            "Bake at 375F (190C) F for 10 to 12 minutes."
        ],
        "link":"recipeland.com\/recipe\/v\/oatmeal-breakfast-cookies-33408",
        "source":"Recipes1M",
    },
...
]

The CaT-Bench benchmark (~20k samples) looks like this:

[
    {
        "plan_idx": 0,
        "title": "spicy-tomato-anchovy-pasta",
        "question_idx": 0,
        "steps": [
            "Heat 6 tablespoons olive oil in a large frying pan over medium heat, then stir in garlic, broccoli and mushrooms;",
            "cook until lightly browned.",
            "Add anchovies and water, cover and simmer for 4 to 5 minutes.",
            "Stir in spring onions, tomatoes and parsley and cover again, simmering until vegetables are soft, about 3 to 4 minutes.",
            "While the vegetables are cooking, bring a large pot of water and one teaspoon of olive oil to the boil.",
            "Add linguine and cook until al dente, about 7 to 8 minutes;",
            "drain.",
            "Toss with anchovy mixture and chilli flakes.",
            "If desired, season with black pepper.",
            "Serve immediately."
        ],
        "question_type": "dependent_real_after",
        "step_pair_idx_asked_about": [
            7,
            9
        ],
        "binary_question": "Must Step 10 happen after Step 8?",
        "why_question": "Why must Step 10 happen after Step 8?",
        "label": 1,
        "type": "real",
        "direction": "after"
    },
...
]

```
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
        
        if prompt_type == 'minimal':
            self.make_fn = self.make_pair_samples_minimal
        elif prompt_type == 'natlang':
            self.make_fn = self.make_pair_samples_natlang
        elif prompt_type == 'only_shuffled':
            self.make_fn = lambda x: self.make_one_sided_samples(x, side='shuf')
        elif prompt_type == 'only_original':
            self.make_fn = lambda x: self.make_one_sided_samples(x, side='orig')
        elif prompt_type == 'pos_neg':
            self.make_fn = self.make_pos_neg_samples
        if batch_mode == 'pos_neg':
            self.make_pos_neg_dataset()
        elif batch_mode == 'random_samples':
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
            print(f"Pruned {removed} sequences exceeding {self.max_length} tokens.")
        self.data = data_pruned

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def make_random_samples_dataset(self):
        formatted_data = []
        for item in tqdm(self.data, desc='Formatting dataset (unbatched)...'):
            formatted_data.append(self.make_fn([item])[0])
        self.data = formatted_data

    def make_pos_neg_dataset(self):
        formatted_data = []
        for batch in tqdm(self.data, desc='Formatting dataset (batched)...'):
            formatted_batch = self.make_fn(batch)
            formatted_data.append(formatted_batch)
        self.data = formatted_data

    def make_pos_neg_samples(self, batch):
        formatted_batch = []
        for i, item in enumerate(batch):
            target_list = item['orig' if item['binary_label'] else 'shuf']
            
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
```

```
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
```

```
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
```

```
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import Seq2SeqDataset, Collator, make_random_samples_dataset, make_pos_neg_samples_dataset
from utils_sys import save_model_tokenizer, setup_config, save_prompt_example
from loss_functions import KLDivergenceLoss, LinearRefinementLoss, CausalLMLoss, MaxMarginLoss
from tqdm.auto import tqdm
import argparse
import json

def main(args):
    train_config = setup_config(args)
    
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = data[:args.num_samples]
    
    if args.batch_mode == 'pos_neg':
        data_pairs = make_pos_neg_samples_dataset(data, k = args.k)
    else:
        data_pairs = make_random_samples_dataset(data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Active Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 output_hidden_states = True
                                                 ).to(device)
    model.train()

    if args.use_kl:
        print(f"Loading Reference Model (Frozen): {args.model_name}")
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        ref_model.eval()
        for param in ref_model.parameters(): # ref model needs to be frozen
            param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    max_length = 2048
    if 'gpt2' in args.model_name:
        max_length = 1024
        tokenizer.add_prefix_space = True
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    
    dataset = Seq2SeqDataset(data_pairs,
                            tokenizer,
                            max_length,
                            prompt_type = args.prompt_type,
                            attention_mask_type = args.attention_mask_type,
                            batch_mode = args.batch_mode,
                            min_recipe_steps = args.min_recipe_steps,
                            ) 
    
    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collator.seq2seq_collate,
                            shuffle=True,
                            )
    
    causal_lm_loss_fn = CausalLMLoss()
    max_margin_loss_fn = MaxMarginLoss(alpha=args.margin_alpha)
    linear_refinement_loss_fn = LinearRefinementLoss()
    kl_loss_fn = KLDivergenceLoss(ref_model) if args.use_kl else None
    
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    tbar = tqdm(dataloader)

    steps = 0

    for batch in tbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        lhs = outputs.hidden_states[-1]

        # --- Loss Computation ---

        # 1. Causal LM Task Loss
        if args.use_causal_lm_loss:
            task_loss = causal_lm_loss_fn(logits, batch['input_ids'], batch['attention_mask'])
        else:
            task_loss = torch.tensor(0.0, device=device)

        # 2. KL Divergence Loss
        if args.use_kl:
            kl_loss = kl_loss_fn(logits, batch['input_ids'], batch['attention_mask'])
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # 3. Geometric / Order Loss
        if args.use_order_loss:
            raw_geo_loss = linear_refinement_loss_fn(lhs, batch['step_indices'])
            order_loss = args.geo_lambda * raw_geo_loss
        else:
            order_loss = torch.tensor(0.0, device=device)

        # 4. Max Margin Loss
        if args.use_max_margin_loss:
            max_margin_loss = max_margin_loss_fn(lhs, batch['step_indices'], batch['binary_label'])
        else:
            max_margin_loss = torch.tensor(0.0, device=device)

        total_loss = task_loss + (args.kl_beta * kl_loss) + order_loss + max_margin_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        tbar.set_description(f'| Causal: {task_loss.item():.3f} '
                             f'| KL: {kl_loss.item():.3f} '
                             f'| Order: {order_loss.item():.3f} '
                             f'| MML: {max_margin_loss.item():.3f}'
                             f'| Batch Size: {batch["input_ids"].shape[0]}'
                             )
        if steps == 0:
            for i in range(batch['input_ids'].shape[0]):
                decoded_all = tokenizer.decode(batch['input_ids'][i])
                mask = batch['attention_mask'][i] == 1
                valid_inputs = batch['input_ids'][i][mask]
                decoded_valid = tokenizer.decode(valid_inputs)
                prompt = decoded_all + '\n\n' + ('-' * 100) + '\n\n' + decoded_valid + '\n\n' + ('#' * 100)
                save_prompt_example(prompt, train_config)
                                
        import pdb; pdb.set_trace()
        steps += 1
        if steps % args.save_interval == 0:
            save_config = train_config.copy()
            save_config['steps'] = steps
            save_model_tokenizer(model, tokenizer, save_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes")
    parser.add_argument("--model_name", help="model path or name", default = "openai-community/gpt2")
    parser.add_argument("--data_path", help="dataset path", default = "./data/recipenlg/recipenlg_clean_100k.json")
    parser.add_argument("--prompt_type", help="type of prompt to use while training", default = "minimal")
    parser.add_argument("--attention_mask_type", help="type of prompt to use while training", default = "full_input")
    parser.add_argument("--num_samples", help="number of samples to draw from the dataset", default = 10_000, type = int)
    parser.add_argument("--batch_size", help="batch size", default = 8, type = int)
    parser.add_argument("--lr", help="learning rate", default = 5e-5, type = float)
    parser.add_argument("--save_interval", help="number of steps after which the model is saved", default = 1000, type = int)
    parser.add_argument("--use_causal_lm_loss", default=0, type=int, help="Enable causal lm loss")
    parser.add_argument("--use_kl", help="whether to include KL normalization in the loss", default = 0, type = int)
    parser.add_argument("--kl_beta", help="KL beta term", default = 0.1, type = float)
    parser.add_argument("--use_order_loss", default=0, type=int, help="Enable geometric refinement loss")
    parser.add_argument("--geo_lambda", default=0.1, type=float, help="Weight for the geometric loss")
    parser.add_argument("--use_max_margin_loss", default=0, type=int, help="Enable geometric refinement loss")
    parser.add_argument("--margin_alpha", help="max margin loss alpha", default = 0.05, type = float)
    parser.add_argument("--batch_mode", default='random_samples', type=int, help="Type of batch to use for training") # `random_samples`, `pos_neg`
    parser.add_argument("--k", default=8, type=int, help="Number of pairs per batch when using batched pair generation")
    parser.add_argument("--min_recipe_steps", default=4, type=int, help="Minimum number of steps for a recipe")
    args = parser.parse_args()

    if args.prompt_type in ['pos_neg', 'only_shuffled', 'only_original']:
        assert args.attention_mask_type == 'full_input', f'Only `args.attention_mask_type` == `full_input` makes sense with `prompt_type` == `{args.prompt_type}`'

    main(args)
```