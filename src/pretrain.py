# src/pretrain.py
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_data import (
    Seq2SeqDataset, Collator, make_random_samples_dataset, make_pos_neg_samples_dataset, prepare_text_batch_prompt
)
from utils_sys import save_run, setup_config
from utils_viz import save_heatmaps
from utils_model import PositionHead
from train.forward import compute_forward_bundle
from train.pos_adv import compute_pos_adv_loss
from train.logging import log_probe_stats
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
    print(f"Train config:\n{json.dumps(train_config, indent=4)}", flush=True)

    with open(args.data_path, "r", encoding="utf8") as f:
        data = json.load(f)
        data = sorted(data, key=lambda x: random.random())

    if args.num_samples > 0:
        data = data[: args.num_samples]

    if args.batch_mode == "pos_neg":
        data_pairs = make_pos_neg_samples_dataset(data, k=args.k)
    elif args.batch_mode == "random_samples":
        data_pairs = make_random_samples_dataset(data, neg_ratio=args.neg_ratio)
        data_pairs = sorted(data_pairs, key=lambda x: random.random())
    else:
        raise ValueError(f"Unknown batch_mode: {args.batch_mode}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Active Model: {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        output_hidden_states=True,
    ).to(device)
    model.train()

    ref_model = None
    if args.use_kl:
        print(f"Loading Reference Model (Frozen): {args.model_name}", flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    max_length = 2048
    if "gpt2" in args.model_name:
        max_length = 1024
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id

    dataset = Seq2SeqDataset(
        data_pairs,
        tokenizer,
        max_length,
        prompt_type=args.prompt_type,
        attn_mask_type=args.attn_mask_type,
        loss_mask_type=args.loss_mask_type,
        batch_mode=args.batch_mode,
        min_recipe_steps=args.min_recipe_steps,
    )

    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator.seq2seq_collate,
        shuffle=True,
    )

    causal_lm_loss_fn = CausalLMLoss()
    max_margin_loss_fn = MaxMarginLoss(alpha=args.margin_alpha, activations=args.activations)
    kl_loss_fn = KLDivergenceLoss(ref_model) if args.use_kl else None

    # Optional position adversary head
    pos_head = None
    if args.use_pos_adv:
        d_model = model.config.n_embd
        pos_head = PositionHead(d_model=d_model, n_bins=args.pos_bins, hidden=args.pos_head_hidden).to(device)
        pos_head.train()

    params = list(model.parameters())
    if pos_head is not None:
        params += list(pos_head.parameters())
    optimizer = AdamW(params=params, lr=args.lr)

    tbar = tqdm(dataloader)

    num_steps = 0
    losses = []
    prompt = None

    for batch in tbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits, lhs, lhs_mml = compute_forward_bundle(args, model, batch)

        # pos-adv uses normal lhs
        pos_loss, pos_acc = (torch.tensor(0.0, device=device), None)
        if args.use_pos_adv:
            if lhs is None:
                raise RuntimeError("use_pos_adv requires normal lhs; check compute_forward_bundle.")
            pos_loss, pos_acc = compute_pos_adv_loss(args, pos_head, lhs, batch)

        if args.save_heatmaps:
            hs = lhs_mml if args.no_pos_mml else lhs
            S_directed, S_undirected = compute_scores(hs[0], batch["step_indices_mml"][0])
            save_heatmaps(S_directed, S_undirected, suffix=f"_{num_steps}")

        loss = gather_losses(
            args,
            causal_lm_loss_fn,
            kl_loss_fn,
            max_margin_loss_fn,
            logits,
            batch,
            device,
            lhs_mml if args.use_mml else lhs,  # safe fallback
            pos_loss,
        )

        optimizer.zero_grad()
        loss["total_loss"].backward()
        optimizer.step()

        log_probe_stats(
            args,
            num_steps,
            float(loss["max_margin_loss"].detach().cpu()),
            float(loss["pos_loss"].detach().cpu()),
            pos_acc.detach().cpu() if pos_acc is not None else None,
        )

        tbar.set_description(
            f"| Causal: {loss['causal_lm_loss'].item():.3f} "
            f"| KL: {loss['kl_loss'].item():.3f} "
            f"| MML: {loss['max_margin_loss'].item():.3f} "
            f"| POS: {loss['pos_loss'].item():.3f} "
        )

        losses.append(
            {
                "step": num_steps,
                "total": float(loss["total_loss"].detach().cpu()),
                "causal": float(loss["causal_lm_loss"].detach().cpu()),
                "kl": float(loss["kl_loss"].detach().cpu()),
                "mml": float(loss["max_margin_loss"].detach().cpu()),
                "pos": float(loss["pos_loss"].detach().cpu()),
            }
        )

        if num_steps == 0:
            prompt = prepare_text_batch_prompt(batch, tokenizer)
            os.makedirs("./misc", exist_ok=True)
            print(prompt, file=open("./misc/last_prompt.txt", "w"), flush=True)

        num_steps += 1
        if num_steps % args.save_interval == 0:
            save_config = train_config.copy()
            save_config["num_steps"] = num_steps
            model_save_dir = os.path.join(train_config["model_save_dir"], str(num_steps))
            save_run(save_config, model_save_dir, model, tokenizer, prompt)

    json_path = os.path.join(train_config["model_save_dir"], "losses.json")
    if os.path.exists(train_config["model_save_dir"]):
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes")
    parser.add_argument("--model_name", default="openai-community/gpt2")
    parser.add_argument("--data_path", default="./data/recipenlg/recipenlg_clean_100k.json")
    parser.add_argument("--batch_mode", default="random_samples", type=str)  # random_samples | pos_neg
    parser.add_argument("--prompt_type", default="minimal_pairs")
    parser.add_argument("--attn_mask_type", default="completion_only")
    parser.add_argument("--loss_mask_type", default="completion_only")
    parser.add_argument("--num_samples", default=10_000, type=int)
    parser.add_argument("--neg_ratio", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)

    parser.add_argument("--use_clm", default=1, type=int)
    parser.add_argument("--clm_lambda", default=1.0, type=float)

    parser.add_argument("--use_kl", default=0, type=int)
    parser.add_argument("--kl_lambda", default=0.1, type=float)

    parser.add_argument("--use_mml", default=0, type=int)
    parser.add_argument("--no_pos_mml", default=0, type=int, help="Use no-pos forward pass for MML hidden states")
    parser.add_argument("--mml_lambda", default=0.1, type=float)
    parser.add_argument("--margin_alpha", default=0.05, type=float)

    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--min_recipe_steps", default=0, type=int)
    parser.add_argument("--save_heatmaps", default=0, type=int)
    parser.add_argument("--activations", default="real", type=str, help="real | non-negative")

    # Positional adversary (GRL)
    parser.add_argument("--use_pos_adv", default=0, type=int)
    parser.add_argument("--pos_lambda", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=5.0, type=float)
    parser.add_argument("--pos_bins", default=32, type=int)
    parser.add_argument("--pos_head_hidden", default=256, type=int)
    parser.add_argument("--log_interval", default=100, type=int)

    args = parser.parse_args()

    if args.prompt_type in ["minimal_mono", "only_shuffled", "only_original"]:
        assert args.attn_mask_type == "full_input", (
            f"Only attn_mask_type==full_input makes sense with prompt_type=={args.prompt_type}"
        )

    assert any([args.use_clm, args.use_kl, args.use_mml]), "at least one loss has to be used!"
    main(args)