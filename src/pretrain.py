import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import Seq2SeqDataset, Collator
from utils_models import prep_inputs_for_causal_lm
import random
from tqdm.auto import tqdm
import os
import argparse
import json
import random

def main(args):
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = data[:args.num_samples]

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
    
    print(f'pos_ratio: {pos_ratio:.6f}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    max_length = 2048
    if 'gpt2' in args.model_name:
        max_length = 1024
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    
    dataset = Seq2SeqDataset(data_pairs,
                            tokenizer,
                            max_length,
                            prompt_type = args.prompt_type,
                            loss_type = args.loss_type,
                            )
    batch_size = 16
    lr = 5e-5
    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator.seq2seq_collate, shuffle=True)

    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr)
    tbar = tqdm(dataloader)

    steps = 0

    for batch in tbar:
        batch = prep_inputs_for_causal_lm(labels=batch['input_ids'],
                                          attention_mask=batch['attention_mask'],
                                          bos_token_id=tokenizer.bos_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        model_outputs = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        logits = model_outputs.logits
        optimizer.zero_grad()

        loss = loss_fn(logits.view(-1, model.config.vocab_size), batch['labels'].view(-1))

        loss.backward()

        optimizer.step()
        tbar.set_description(f'{loss.item():.3f}')

        steps += 1

        if steps % args.save_interval == 0:
            model_name_simple = args.model_name.split('/')[-1]
            model_save_dir = os.path.join(f'./models/recipenlg/{args.prompt_type}/{args.loss_type}', f"{model_name_simple}_{steps}")
            os.makedirs(model_save_dir, exist_ok=True)
            with open(os.path.join(model_save_dir, 'config.json'), 'w', encoding='utf8') as f:
                json.dump(args.__dict__, f, ensure_ascii = False, indent = 4)
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes")
    parser.add_argument("--model_name", help="model path or name", default = "openai-community/gpt2")
    parser.add_argument("--data_path", help="dataset path", default = "./data/recipenlg/recipenlg_processed.json")
    parser.add_argument("--prompt_type", help="type of prompt to use while training", default = "minimal")
    parser.add_argument("--loss_type", help="type of prompt to use while training", default = "full")
    parser.add_argument("--num_samples", help="number of samples to draw from the dataset", default = 10_000, type = int)
    parser.add_argument("--batch_size", help="number of samples to draw from the dataset", default = 16, type = int)
    parser.add_argument("--save_interval", help="number of steps after which the model is saved", default = 1000, type = int)
    parser.add_argument("--lr", help="number of samples to draw from the dataset", default = 5e-5, type = float)
    args = parser.parse_args()
    main(args)