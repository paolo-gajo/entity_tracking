import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import Seq2SeqDataset, Collator, make_pairs_from_recipenlg
from utils_sys import save_model_tokenizer
from tqdm.auto import tqdm
import os
import argparse
import json

def main(args):
    train_config = args.__dict__
    with open(train_config['data_path'], 'r', encoding='utf8') as f:
        data = json.load(f)

    if train_config['num_samples'] > 0:
        data = data[:train_config['num_samples']]

    data_pairs = make_pairs_from_recipenlg(data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Active Model: {train_config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(train_config['model_name']).to(device)
    model.train()

    if train_config['use_kl']:
        print(f"Loading Reference Model (Frozen): {train_config['model_name']}")
        ref_model = AutoModelForCausalLM.from_pretrained(train_config['model_name']).to(device)
        ref_model.eval()
        for param in ref_model.parameters(): # ref model needs to be frozen
            param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(train_config['model_name'])

    max_length = 2048
    if 'gpt2' in train_config['model_name']:
        max_length = 1024
        tokenizer.add_prefix_space = True
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    
    dataset = Seq2SeqDataset(data_pairs,
                            tokenizer,
                            max_length,
                            prompt_type = train_config['prompt_type'],
                            loss_type = train_config['loss_type'],
                            ) 
    
    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], collate_fn=collator.seq2seq_collate, shuffle=False)
    
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=train_config['lr'])
    tbar = tqdm(dataloader)

    steps = 0

    for batch in tbar:
        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][batch['attention_mask'] == 0] = -100
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        
        shift_labels = batch['labels'][..., 1:].contiguous()
        shift_attention_mask = batch['attention_mask'][..., 1:].contiguous()

        task_loss = loss_fn(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))

        if train_config['use_kl']:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                ref_logits = ref_outputs.logits
    
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    
            log_probs_model = F.log_softmax(shift_logits, dim=-1)
            probs_ref = F.softmax(shift_ref_logits, dim=-1)
        
            kl_per_token = F.kl_div(log_probs_model, probs_ref, reduction='none').sum(dim=-1)
            valid_mask = shift_attention_mask.float()
            
            num_valid_tokens = valid_mask.sum()
            if num_valid_tokens > 0:
                kl_loss = (kl_per_token * valid_mask).sum() / num_valid_tokens
            else:
                kl_loss = torch.tensor(0.0, device=device)
        else:
            kl_loss = torch.tensor(0.0, device=device)

        total_loss = task_loss + (args.kl_beta * kl_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        tbar.set_description(f'Task: {task_loss.item():.3f} | KL: {kl_loss.item():.3f}')

        steps += 1
        
        if steps % train_config['save_interval'] == 0:
            train_config['steps'] = steps
            save_model_tokenizer(model, tokenizer, train_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes")
    parser.add_argument("--model_name", help="model path or name", default = "openai-community/gpt2")
    parser.add_argument("--data_path", help="dataset path", default = "./data/recipenlg/recipenlg_processed.json")
    parser.add_argument("--prompt_type", help="type of prompt to use while training", default = "minimal")
    parser.add_argument("--loss_type", help="type of prompt to use while training", default = "full")
    parser.add_argument("--num_samples", help="number of samples to draw from the dataset", default = 10_000, type = int)
    parser.add_argument("--batch_size", help="batch size", default = 16, type = int)
    parser.add_argument("--use_kl", help="whether to include KL normalization in the loss", default = 0, type = int)
    parser.add_argument("--kl_beta", help="KL beta term", default = 0.1, type = float)
    parser.add_argument("--save_interval", help="number of steps after which the model is saved", default = 1000, type = int)
    parser.add_argument("--lr", help="learning rate", default = 5e-5, type = float)
    args = parser.parse_args()
    main(args)