import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from utils_data import (
    Seq2SeqDataset, Collator, make_random_samples_dataset,
    make_pos_neg_samples_dataset, prepare_text_batch_prompt,
)
from utils_sys import save_run, setup_config, capture_source_snapshot
from utils_viz import save_heatmaps
from utils_model import PositionHead, build_model_tokenizer
from train.forward import compute_forward_bundle
from train.pos_adv import compute_pos_adv_loss
from train.logging import log_probe_stats
from sims import compute_scores
from loss_functions import (
    KLDivergenceLoss,
    CausalLMLoss,
    MaxMarginLoss,
    CosineContrastiveLoss,
    StepTokenLoss,
    gather_losses,
)
from tqdm.auto import tqdm
import argparse
import json
import random
import os

torch.set_printoptions(linewidth=100000)

def main(args):
    source_snapshot = capture_source_snapshot()
    resume_steps = 0
    if args.resume_from:
        ckpt_config_path = os.path.join(args.resume_from, "train_config.json")
        with open(ckpt_config_path, "r", encoding="utf8") as f:
            ckpt_config = json.load(f)
        resume_steps = ckpt_config["num_steps"]
        print(f"Resuming from checkpoint: {args.resume_from} (step {resume_steps})", flush=True)
        args.model_name = args.resume_from

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
    tokenizer, step_token_id_map, model, ref_model = build_model_tokenizer(args, device)

    dataset = Seq2SeqDataset(
        data_pairs,
        tokenizer,
        max_length=model.config.max_position_embeddings,
        prompt_type_list=train_config['prompt_type_list'],
        attn_mask_type=args.attn_mask_type,
        clm_mask_type=args.clm_mask_type,
        batch_mode=args.batch_mode,
        min_recipe_steps=args.min_recipe_steps,
        max_recipe_steps=args.stp_max_steps,
        step_token_id_map=step_token_id_map,
        prepend_bos=False,
    )    

    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator.seq2seq_collate,
        shuffle=True,
    )

    # ---- Loss functions ---------------------------------------------------

    # CLM loss
    causal_lm_loss_fn = CausalLMLoss()

    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.n_embd
    max_margin_loss_fn = MaxMarginLoss(
        alpha=args.margin_alpha,
        activations=args.activations,
        hidden_dim=hidden_dim,
        proj_dim=args.mml_proj_dim,
    ).to(device)
    cos_loss_fn = CosineContrastiveLoss(alpha=args.cos_alpha) if args.use_cos else None
    kl_loss_fn = KLDivergenceLoss(ref_model) if args.use_kl else None

    # ---- Step Token Prediction loss ----------------------------------------
    stp_loss_fn = None
    if args.use_stp:
        stp_loss_fn = StepTokenLoss().to(device)

    # ---- Position adversary head ------------------------------------------
    pos_head = None
    if args.use_grl:
        d_model = model.config.n_embd
        pos_head = PositionHead(d_model=d_model, n_bins=args.pos_bins, hidden=args.pos_head_hidden).to(device)
        pos_head.train()

    # ---- Optimizer --------------------------------------------------------
    params = list(model.parameters())
    if pos_head is not None:
        params += list(pos_head.parameters())
    if max_margin_loss_fn.proj is not None:
        params += list(max_margin_loss_fn.proj.parameters())
    optimizer = AdamW(params=params, lr=args.lr)

    tbar = tqdm(dataloader)
    num_steps = resume_steps
    losses = []
    prompt = None

    amp_dtype = getattr(torch, getattr(args, 'dtype', 'float32'))
    use_amp = amp_dtype != torch.float32

    for batch_idx, batch in enumerate(tbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, lhs = compute_forward_bundle(args, model, batch)

            if logits is not None and torch.isnan(logits).any():
                print("NaN in logits before loss")

            # ---- STP loss ---------------------------------------------------
            stp_loss = torch.tensor(0.0, device=device)
            if args.use_stp and stp_loss_fn is not None:
                stp_loss = stp_loss_fn(logits, batch['input_ids'], batch['stp_mask'])

            # ---- Pos-adv loss -----------------------------------------------
            pos_loss, pos_acc = (torch.tensor(0.0, device=device), None)
            if args.use_grl:
                if lhs is None:
                    raise RuntimeError("use_grl requires normal lhs; check compute_forward_bundle.")
                pos_loss, pos_acc = compute_pos_adv_loss(args, pos_head, lhs, batch)

            if args.save_heatmaps:
                S_directed, S_undirected = compute_scores(lhs[0], batch["step_indices"][0])
                save_heatmaps(S_directed, S_undirected, suffix=f"_{num_steps}")

            # ---- Pooled CLM: pass completion_step_indices if available -------
            # For the pooled variant, gather_losses needs step_indices to
            # actually contain the *completion-side* step indices.  When using
            # prompt_type='pooled_pairs', the collator produces a separate
            # 'completion_step_indices' tensor.  We temporarily swap it in.
            if args.pool_clm and 'completion_step_indices' in batch:
                original_mml = batch.get('step_indices')
                batch['step_indices'] = batch['completion_step_indices']

            loss = gather_losses(
                args,
                causal_lm_loss_fn,
                kl_loss_fn,
                max_margin_loss_fn,
                logits,
                batch,
                device,
                lhs,
                pos_loss,
                stp_loss,
                cos_loss_fn=cos_loss_fn,
            )

        # Restore original MML indices if swapped
        if args.pool_clm and 'completion_step_indices' in batch:
            batch['step_indices'] = original_mml

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

        stp_val = float(loss['stp_loss'].detach().cpu()) if 'stp_loss' in loss else 0.0

        cos_val = float(loss['cos_loss'].detach().cpu()) if 'cos_loss' in loss else 0.0

        tbar.set_description(
            f"| CLM: {loss['causal_lm_loss'].item():.3f} "
            f"| KL: {loss['kl_loss'].item():.3f} "
            f"| MML: {loss['max_margin_loss'].item():.3f} "
            f"| COS: {cos_val:.3f} "
            f"| POS: {loss['pos_loss'].item():.3f} "
            f"| STP: {stp_val:.3f} "
        )

        losses.append(
            {
                "step": num_steps,
                "total": float(loss["total_loss"].detach().cpu()),
                "causal": float(loss["causal_lm_loss"].detach().cpu()),
                "kl": float(loss["kl_loss"].detach().cpu()),
                "mml": float(loss["max_margin_loss"].detach().cpu()),
                "cos": cos_val,
                "pos": float(loss["pos_loss"].detach().cpu()),
                "stp": stp_val,
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
            save_run(save_config, model_save_dir, model, tokenizer, prompt, source_snapshot)
        
    json_path = os.path.join(train_config["model_save_dir"], "losses.json")
    if os.path.exists(train_config["model_save_dir"]):
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train a causal LM on RecipeNLG to learn to unshuffle recipes"
    )
    parser.add_argument("--model_name", default="openai-community/gpt2")
    parser.add_argument("--resume_from", default=None, type=str,
                        help="Path to a checkpoint directory to resume from. "
                             "Reads train_config.json to get the step count and loads the model from that directory.")
    parser.add_argument("--data_path", default="./data/recipenlg/recipenlg_clean.json")
    parser.add_argument("--batch_mode", default="random_samples", type=str)
    parser.add_argument("--prompt_type", default="minimal_pairs")
    parser.add_argument("--attn_mask_type", default="full")
    parser.add_argument("--clm_mask_type", default="completion_only")
    parser.add_argument("--num_samples", default=1_000_000, type=int)
    parser.add_argument("--neg_ratio", default=0.5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    
    # Absolute positional embeddings
    parser.add_argument("--use_abs_pe", default=0, type=int,
                        help="Inject learned absolute positional embeddings into the model")
    parser.add_argument("--abs_pe_max_len", default=1024, type=int,
                        help="Max sequence length for absolute positional embeddings")

    # PEFT
    parser.add_argument("--use_lora", default=0, type=int)

    # Causal LM loss
    parser.add_argument("--use_clm", default=0, type=int)
    parser.add_argument("--clm_lambda", default=1.0, type=float)
    parser.add_argument("--pool_clm", default=0, type=int, help="Use per-step pooled CLM loss (Section 3.1)")

    # KL
    parser.add_argument("--use_kl", default=0, type=int)
    parser.add_argument("--kl_lambda", default=0.1, type=float)

    # Max-margin loss
    parser.add_argument("--use_mml", default=0, type=int)
    parser.add_argument("--no_pos_mml", default=0, type=int, help="Use no-pos forward pass for MML hidden states")
    parser.add_argument("--mml_lambda", default=0.1, type=float)
    parser.add_argument("--margin_alpha", default=0.05, type=float)
    parser.add_argument("--mml_proj_dim", default=0, type=int,
                        help="If > 0, apply MML on a learned projection of this dim instead of raw hidden states")

    # Cosine contrastive loss
    parser.add_argument("--use_cos", default=0, type=int)
    parser.add_argument("--cos_lambda", default=0.1, type=float)
    parser.add_argument("--cos_alpha", default=0.5, type=float,
                        help="Negative-pair margin: penalise cos_sim > cos_alpha")

    # Step Token Prediction (Section 3.2)
    parser.add_argument("--use_stp", default=0, type=int,
                        help="Enable step token prediction loss")
    parser.add_argument("--stp_lambda", default=1.0, type=float,
                        help="Weight for step token prediction loss")
    parser.add_argument("--stp_max_steps", default=15, type=int,
                        help="M: number of step tokens added to the vocabulary")
    parser.add_argument("--init_from_eos", default=0, type=int,
                        help="Whether to initialize the new step token embeddings with the same values as the EOS token embedding")

    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--min_recipe_steps", default=0, type=int)
    parser.add_argument("--save_heatmaps", default=0, type=int)
    parser.add_argument("--activations", default="real", type=str, help="real | non-negative")

    # Positional adversary (GRL)
    parser.add_argument("--use_grl", default=0, type=int)
    parser.add_argument("--pos_lambda", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=5.0, type=float)
    parser.add_argument("--pos_bins", default=32, type=int)
    parser.add_argument("--pos_head_hidden", default=256, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--dtype", default="bfloat16", type=str,
                        help="Model dtype: float32, bfloat16, float16")
    parser.add_argument("--revision", default=None, type=str,
                        help="Model revision/checkpoint to load (e.g. 'step4000' for Pythia early checkpoints)")

    args = parser.parse_args()

    # ---- Validation -------------------------------------------------------
    assert args.attn_mask_type == "full", "Only attn_mask_type==full makes sense"

    if args.use_stp:
        assert "step_token_pairs" in args.prompt_type.split('+'), (
            f"use_stp requires prompt_type='step_token_pairs', got '{args.prompt_type}'"
        )

    assert any([args.use_clm, args.use_kl, args.use_mml, args.use_cos, args.use_stp]), (
        "at least one loss has to be used!"
    )

    main(args)
