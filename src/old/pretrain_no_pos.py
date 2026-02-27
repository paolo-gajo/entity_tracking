# pretrain.py
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import Seq2SeqDataset, Collator, make_random_samples_dataset, make_pos_neg_samples_dataset, prepare_text_batch_prompt
from utils_sys import save_run, get_current_time_string, setup_config
from utils_viz import save_heatmaps
from utils_model import forward_no_pos_gpt2
from sims import compute_scores
from loss_functions import KLDivergenceLoss, CausalLMLoss, MaxMarginLoss, gather_losses
from tqdm.auto import tqdm
import argparse
import json
import random
import os

torch.set_printoptions(linewidth=100000)

def main(args):
    train_config = setup_config(args.__dict__)
    print(f'Train config:\n{json.dumps(train_config, indent = 4)}')
    
    with open(args.data_path, 'r', encoding='utf8') as f:
        data = json.load(f)
        data = sorted(data, key = lambda x: random.random())

    if args.num_samples > 0:
        data = data[:args.num_samples]
    
    if args.batch_mode == 'pos_neg':
        data_pairs = make_pos_neg_samples_dataset(data, k = args.k)
    elif args.batch_mode == 'random_samples':
        data_pairs = make_random_samples_dataset(data, neg_ratio = args.neg_ratio)
        data_pairs = sorted(data_pairs, key = lambda x: random.random())

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space = True)

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
                            attn_mask_type = args.attn_mask_type,
                            loss_mask_type = args.loss_mask_type,
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
    max_margin_loss_fn = MaxMarginLoss(alpha=args.margin_alpha, activations=args.activations)
    kl_loss_fn = KLDivergenceLoss(ref_model) if args.use_kl else None
    
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    tbar = tqdm(dataloader)

    num_steps = 0
    losses = []
    prompt = None
    for batch in tbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = None
        lhs = None

        if (args.use_clm or
            args.use_kl or
            (args.save_heatmaps and not args.no_pos_mml) or
            (args.use_mml and not args.no_pos_mml)
            ):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attn_mask'])
            logits = outputs.logits
            lhs = outputs.hidden_states[-1]

        if args.no_pos_mml:
            out_np = forward_no_pos_gpt2(
                model,
                input_ids=batch['input_ids'],
                attention_mask=batch['attn_mask'],
                output_hidden_states=True,
            )
            lhs_mml = out_np.last_hidden_state  # [B, T, D]
        else:
            lhs_mml = lhs
        import pdb; pdb.set_trace()
        if args.save_heatmaps:
            hs = lhs_mml if args.no_pos_mml else lhs
            S_directed, S_undirected = compute_scores(hs[0], batch['step_indices_mml'][0])
            save_heatmaps(S_directed, S_undirected, suffix=f'_{num_steps}')
        loss = gather_losses(args, causal_lm_loss_fn, kl_loss_fn, max_margin_loss_fn, logits, batch, device, lhs_mml)
        optimizer.zero_grad()
        loss['total_loss'].backward()
        optimizer.step()
        tbar.set_description(f"| Causal: {loss['causal_lm_loss'].item():.3f} "
                             f"| KL: {loss['kl_loss'].item():.3f} "
                             f"| MML: {loss['max_margin_loss'].item():.3f} "
                             )

        losses.append({
            "step": num_steps,
            "total": float(loss["total_loss"].detach().cpu()),
            "causal": float(loss["causal_lm_loss"].detach().cpu()),
            "kl": float(loss["kl_loss"].detach().cpu()),
            "mml": float(loss["max_margin_loss"].detach().cpu()),
        })

        if num_steps == 0:
            prompt = prepare_text_batch_prompt(batch, tokenizer)
            print(prompt, file = open('./misc/last_prompt.txt', 'w'))

        num_steps += 1
        if num_steps % args.save_interval == 0:
            save_config = train_config.copy()
            save_config['num_steps'] = num_steps
            model_save_dir = os.path.join(train_config['model_save_dir'], str(num_steps))
            save_run(save_config, model_save_dir, model, tokenizer, prompt)
    json_path = os.path.join(train_config['model_save_dir'], 'losses.json')
    if os.path.exists(train_config['model_save_dir']):
        with open(json_path, 'w', encoding='utf8') as f:
            json.dump(losses, f, ensure_ascii = False, indent = 4) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes")
    parser.add_argument("--model_name", help="model path or name", default = "openai-community/gpt2")
    parser.add_argument("--data_path", help="dataset path", default = "./data/recipenlg/recipenlg_clean_100k.json")
    parser.add_argument("--batch_mode", help="Type of batch to use for training", default='random_samples', type=str) # `random_samples`, `pos_neg`
    parser.add_argument("--prompt_type", help="type of prompt to use while training", default = "minimal_pairs")
    parser.add_argument("--attn_mask_type", help="portion of the input with model attention", default = "completion_only")
    parser.add_argument("--loss_mask_type", help="portion of the input where we calculate causal LM loss", default = "completion_only")
    parser.add_argument("--num_samples", help="number of samples to draw from the dataset", default = 10_000, type = int)
    parser.add_argument("--neg_ratio", help="number of samples to draw from the dataset", default = 0.1, type = float)
    parser.add_argument("--batch_size", help="batch size", default = 8, type = int)
    parser.add_argument("--lr", help="learning rate", default = 5e-5, type = float)
    parser.add_argument("--save_interval", help="number of steps after which the model is saved", default = 1000, type = int)
    parser.add_argument("--use_clm", default=1, type=int, help="Enable causal lm loss")
    parser.add_argument("--clm_lambda", help="lambda hyperparameter for causal LM loss", default = 1.0, type = float)
    parser.add_argument("--use_kl", help="whether to include KL normalization in the loss", default = 0, type = int)
    parser.add_argument("--kl_lambda", help="lambda hyperparameter for KL div", default = 0.1, type = float)
    parser.add_argument("--use_mml", default=0, type=int, help="Enable geometric refinement loss")
    parser.add_argument("--no_pos_mml", default=0, type=int, help="whether to do a no-pos-embedding forward pass for the LHS used for MML loss")
    parser.add_argument("--mml_lambda", help="lambda hyperparameter for max-margin loss", default = 0.1, type = float)
    parser.add_argument("--margin_alpha", help="max margin loss alpha", default = 0.05, type = float)
    parser.add_argument("--k", default=8, type=int, help="Number of pairs per batch when using batched pair generation")
    parser.add_argument("--min_recipe_steps", default=0, type=int, help="Minimum number of steps for a recipe")
    parser.add_argument("--save_heatmaps", default=0, type=int, help="whether to save heatmaps at each step for debugging")
    parser.add_argument("--activations", default='real', type=str, help="whether to force activations to be `non-negative` or just `real`")
    args = parser.parse_args()

    if args.prompt_type in ['minimal_mono', 'only_shuffled', 'only_original']:
        assert args.attn_mask_type == 'full_input', f'Only `args.attn_mask_type` == `full_input` makes sense with `prompt_type` == `{args.prompt_type}`'
    
    assert any([args.use_clm, args.use_kl, args.use_mml]), 'at least one loss has to be used!'

    main(args)