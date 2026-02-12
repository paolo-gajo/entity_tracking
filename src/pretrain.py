import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import Seq2SeqDataset, Collator, make_pairs_from_recipenlg
from utils_sys import save_model_tokenizer
from tqdm.auto import tqdm
import argparse
import json

def compute_linear_refinement_loss(hidden_states, step_ids):
    """
    Enforces H_{t+1} <= H_{t} (Refinement) for the linear sequence 1->2->3...
    This assumes the input 'hidden_states' corresponds to the SORTED recipe.
    """
    # hidden_states: (Batch, Seq_Len, Dim)
    # step_ids: (Batch, Seq_Len) - 0=Pad, 1=Step1, 2=Step2...
    
    loss = torch.tensor(0.0, device=hidden_states.device)
    valid_samples = 0

    for b in range(hidden_states.size(0)):
        # Get the unique steps in order (e.g., [1, 2, 3, 4, 5])
        # We assume the target is the sorted recipe, so steps are monotonic
        steps = step_ids[b].unique()
        steps = steps[steps != 0] # Remove padding
        steps = sorted(steps.tolist())
        
        if len(steps) < 2: continue
            
        # Pool embeddings for each step
        h_list = []
        for s in steps:
            mask = (step_ids[b] == s)
            # Mean pool the tokens for this step
            h_list.append(hidden_states[b][mask].mean(dim=0))
            
        # Stack: (Num_Steps, Dim)
        H = torch.stack(h_list)
        
        # We want H[t+1] inside H[t].
        # Violation = H[t+1] - H[t] > 0
        # Calculate diff between adjacent steps: H[1:] - H[:-1]
        # i.e., Step2 - Step1, Step3 - Step2...
        diff = H[1:] - H[:-1] 
        
        # Penalty: || ReLU(Next - Prev) ||^2
        penalty = torch.norm(torch.relu(diff), dim=-1).pow(2).mean()
        
        loss += penalty
        valid_samples += 1
        
    return loss / max(valid_samples, 1)

def main(args):
    train_config = args.__dict__

    print('Train config:\n')
    print(train_config)
    with open(train_config['data_path'], 'r', encoding='utf8') as f:
        data = json.load(f)

    if train_config['num_samples'] > 0:
        data = data[:train_config['num_samples']]

    data_pairs = make_pairs_from_recipenlg(data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Active Model: {train_config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(train_config['model_name'],
                                                 output_hidden_states = True
                                                 ).to(device)
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
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], collate_fn=collator.seq2seq_collate, shuffle=True)
    
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
        lhs = outputs.hidden_states[-1]
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

        if args.use_order_loss:
            # Pass lhs (Last Hidden State) and indices
            raw_geo_loss = compute_linear_refinement_loss(lhs, batch['step_indices'])
            # Scale it!
            order_loss = args.geo_lambda * raw_geo_loss
        else:
            order_loss = torch.tensor(0.0, device=device)

        total_loss = task_loss + (args.kl_beta * kl_loss) + order_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        tbar.set_description(f'Task: {task_loss.item():.3f} | KL: {kl_loss.item():.3f} | Order: {order_loss.item():.3f}')

        if steps == 0:
            i = 0
            decoded_all = tokenizer.decode(batch['input_ids'][i])
            print(decoded_all)
            print('#' * 100)
            mask = batch['attention_mask'][i] == 1
            valid_inputs = batch['input_ids'][i][mask]
            decoded_valid = tokenizer.decode(valid_inputs)
            print(decoded_valid)
            print('#' * 100)
        steps += 1
        
        if steps % train_config['save_interval'] == 0:
            save_config = train_config.copy()
            if args.use_order_loss:
                save_config['loss_type'] = save_config['loss_type'] + '_with_order_loss'
            save_config['steps'] = steps
            save_model_tokenizer(model, tokenizer, save_config)

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
    parser.add_argument("--use_order_loss", default=0, type=int, help="Enable geometric refinement loss")
    parser.add_argument("--geo_lambda", default=0.1, type=float, help="Weight for the geometric loss")
    args = parser.parse_args()
    main(args)