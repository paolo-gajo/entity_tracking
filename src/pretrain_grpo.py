"""
GRPO training for causal step ordering with Qwen3 models.

Given shuffled recipe steps with step tokens (pi_shuf), the model learns to:
1. Reason about step dependencies inside <think>...</think>
2. Output step tokens in the correct chronological order

Uses Group Relative Policy Optimization (GRPO) with verifiable rewards.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from utils_sys import save_run, setup_config
from utils_data import make_random_samples_dataset
from tqdm.auto import tqdm
import argparse
import json
import random
import os
import re
import math

torch.set_printoptions(linewidth=100000)


# ======================================================================
# Prompt construction
# ======================================================================

def build_grpo_prompt(item, step_token_names, tokenizer):
    """
    Build a chat-formatted prompt for GRPO training.

    Given a shuffled recipe, create a prompt asking the model to reason
    about the correct order and output step tokens accordingly.

    Returns:
        prompt_text: str
        ground_truth_order: list[int] - correct step token indices in orig order
    """
    shuf = item['shuf']
    orig = item['orig']
    n_steps = len(shuf)

    # Build the shuffled steps with step tokens
    steps_text = ""
    for j, step in enumerate(shuf):
        steps_text += f"{step_token_names[j]} {step.strip()}\n"

    # Ground truth: for each position in orig, find which step token (shuf index) it had
    ground_truth_order = []
    for orig_step in orig:
        shuf_idx = shuf.index(orig_step)
        ground_truth_order.append(shuf_idx)

    system_msg = (
        "You are given recipe steps in a shuffled order. Each step is labeled with a step token "
        "(e.g., <step_0>, <step_1>, etc.). Your task is to reason about the correct chronological "
        "order of the steps, then output the step tokens in the correct order.\n\n"
        "First think step-by-step inside <think>...</think> tags, then output ONLY the step tokens "
        "in the correct order, separated by spaces."
    )

    user_msg = f"{steps_text}\nOutput the step tokens in the correct chronological order."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    return prompt_text, ground_truth_order


# ======================================================================
# Response parsing & reward
# ======================================================================

def parse_step_tokens_from_response(response_text, n_steps):
    """
    Parse step token indices from model response.
    Looks for <step_N> patterns after </think> tag.

    Returns:
        list[int] or None if parsing fails
    """
    # Try to find content after </think>
    think_end = response_text.find("</think>")
    if think_end != -1:
        answer_part = response_text[think_end + len("</think>"):]
    else:
        answer_part = response_text

    pattern = r"<step_(\d+)>"
    matches = re.findall(pattern, answer_part)

    if not matches:
        return None

    predicted = [int(m) for m in matches]
    return predicted


def compute_reward(predicted_order, ground_truth_order, n_steps):
    """
    Compute reward for a predicted step ordering.

    Components:
    1. Format reward: +0.1 if response contains the right number of valid step tokens
    2. Kendall tau: pairwise ordering correlation, rescaled to [0, 1]
    3. Exact match bonus: +0.5 if order is exactly correct

    Returns:
        float reward in [0, ~1.6]
    """
    if predicted_order is None:
        return 0.0

    reward = 0.0

    # Format reward: correct number of unique, valid step tokens
    if (len(predicted_order) == n_steps
            and len(set(predicted_order)) == n_steps
            and all(0 <= t < n_steps for t in predicted_order)):
        reward += 0.1
    else:
        return 0.05 if len(predicted_order) > 0 else 0.0

    # Kendall tau correlation
    pred_pos = {v: k for k, v in enumerate(predicted_order)}
    true_pos = {v: k for k, v in enumerate(ground_truth_order)}
    concordant = 0
    discordant = 0
    tokens = list(range(n_steps))
    for i in range(n_steps):
        for j in range(i + 1, n_steps):
            ti, tj = tokens[i], tokens[j]
            if ti in pred_pos and tj in pred_pos:
                pred_diff = pred_pos[ti] - pred_pos[tj]
                true_diff = true_pos[ti] - true_pos[tj]
                if pred_diff * true_diff > 0:
                    concordant += 1
                elif pred_diff * true_diff < 0:
                    discordant += 1

    n_pairs = n_steps * (n_steps - 1) / 2
    if n_pairs > 0:
        tau = (concordant - discordant) / n_pairs
        reward += (tau + 1) / 2  # Rescale [-1,1] -> [0,1]

    # Exact match bonus
    if predicted_order == ground_truth_order:
        reward += 0.5

    return reward


# ======================================================================
# Log-probability helpers
# ======================================================================

def compute_log_probs(model, input_ids, attention_mask, response_start_idx):
    """Compute per-token log probs for the response portion (no grad)."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = log_probs.gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    response_mask = torch.zeros_like(token_log_probs)
    response_mask[:, response_start_idx - 1:] = 1.0

    return token_log_probs, response_mask


def compute_log_probs_with_grad(model, input_ids, attention_mask, response_start_idx):
    """Compute per-token log probs for the response portion (with grad)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = log_probs.gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    response_mask = torch.zeros_like(token_log_probs)
    response_mask[:, response_start_idx - 1:] = 1.0

    return token_log_probs, response_mask


# ======================================================================
# GRPO step
# ======================================================================

def grpo_step(model, ref_model, tokenizer, prompts_data, device, args,
              step_token_names):
    """
    One GRPO update step.

    For each prompt:
    1. Generate G completions from current policy
    2. Compute verifiable rewards
    3. Compute group-relative advantages
    4. Compute clipped policy gradient loss + KL penalty
    """
    G = args.num_generations
    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0
    n_prompts = 0

    for item in prompts_data:
        prompt_text, ground_truth_order = build_grpo_prompt(
            item, step_token_names, tokenizer
        )
        n_steps = len(item['shuf'])

        prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_enc['input_ids'].to(device)
        prompt_len = prompt_ids.shape[1]

        if prompt_len > args.max_prompt_length:
            continue

        # ---- Generate G completions (no grad) --------------------------------
        completions_ids = []
        rewards = []

        model.eval()
        with torch.no_grad():
            for g in range(G):
                output_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                )

                response_ids = output_ids[0, prompt_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                completions_ids.append(output_ids)

                predicted_order = parse_step_tokens_from_response(response_text, n_steps)
                reward = compute_reward(predicted_order, ground_truth_order, n_steps)
                rewards.append(reward)

        if not rewards:
            continue

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        total_reward += rewards_t.mean().item()
        n_prompts += 1

        # ---- Group-relative advantages ---------------------------------------
        if rewards_t.std() > 1e-8:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards_t)

        # ---- Old log probs (from sampling policy) ----------------------------
        old_log_probs_list = []
        response_masks_list = []
        model.eval()
        with torch.no_grad():
            for g in range(G):
                full_ids = completions_ids[g].to(device)
                attn = torch.ones_like(full_ids)
                lp, rm = compute_log_probs(model, full_ids, attn, prompt_len)
                old_log_probs_list.append(lp)
                response_masks_list.append(rm)

        # ---- Policy gradient (with grad) -------------------------------------
        model.train()
        prompt_loss = torch.tensor(0.0, device=device)

        for g in range(G):
            full_ids = completions_ids[g].to(device)
            attn = torch.ones_like(full_ids)

            new_lp, resp_mask = compute_log_probs_with_grad(
                model, full_ids, attn, prompt_len
            )
            old_lp = old_log_probs_list[g].detach()

            # Importance sampling ratio
            ratio = torch.exp(new_lp - old_lp)

            # Clipped surrogate objective
            adv = advantages[g].to(device)
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
            ) * adv

            token_loss = -torch.min(surr1, surr2)
            masked_loss = (token_loss * resp_mask).sum() / (resp_mask.sum() + 1e-8)

            # KL penalty against reference model
            kl_loss = torch.tensor(0.0, device=device)
            if ref_model is not None and args.kl_beta > 0:
                with torch.no_grad():
                    ref_logits = ref_model(
                        input_ids=full_ids, attention_mask=attn
                    ).logits
                    ref_lp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                    ref_token_lp = ref_lp.gather(
                        2, full_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                # KL(π_θ || π_ref) ≈ Σ_t [log π_θ(t) - log π_ref(t)]
                kl_per_token = new_lp - ref_token_lp
                kl_loss = (kl_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)

            prompt_loss = prompt_loss + masked_loss + args.kl_beta * kl_loss

        prompt_loss = prompt_loss / G
        total_loss = total_loss + prompt_loss

    if n_prompts > 0:
        total_loss = total_loss / n_prompts
        avg_reward = total_reward / n_prompts
    else:
        avg_reward = 0.0

    return total_loss, avg_reward


# ======================================================================
# Data preparation
# ======================================================================

def make_shuffled_dataset(data, min_steps=3, max_steps=15):
    """
    Create a dataset of (orig, shuf) pairs where shuf is always a
    random permutation != orig.
    """
    step_list_orig = [
        x['directions'] for x in data
        if len(set(x['directions'])) > 1
        and min_steps <= len(x['directions']) <= max_steps
    ]
    print(f"Filtered dataset: {len(step_list_orig)} recipes "
          f"({min_steps}–{max_steps} steps)")

    pairs = []
    for orig in step_list_orig:
        n = len(orig)
        shuf = random.sample(orig, n)
        attempts = 0
        while shuf == orig and attempts < 100:
            shuf = random.sample(orig, n)
            attempts += 1
        if shuf != orig:
            pairs.append({'orig': orig, 'shuf': shuf, 'binary_label': 0})
    return pairs


# ======================================================================
# Main
# ======================================================================

def main(args):
    train_config = setup_config(args.__dict__)
    print(f"Train config:\n{json.dumps(train_config, indent=4)}", flush=True)

    # Load data
    with open(args.data_path, "r", encoding="utf8") as f:
        data = json.load(f)
        data = sorted(data, key=lambda x: random.random())

    if args.num_samples > 0:
        data = data[:args.num_samples]

    data_pairs = make_shuffled_dataset(
        data,
        min_steps=args.min_recipe_steps,
        max_steps=args.stp_max_steps,
    )
    random.shuffle(data_pairs)
    print(f"Training pairs: {len(data_pairs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Model & tokenizer ------------------------------------------------
    print(f"Loading model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    ).to(device)

    # Add step tokens to vocabulary
    step_token_names = [f"<step_{i}>" for i in range(args.stp_max_steps)]
    tokenizer.add_tokens(step_token_names, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    step_token_id_map = {
        i: tokenizer.convert_tokens_to_ids(f"<step_{i}>")
        for i in range(args.stp_max_steps)
    }

    # Reference model for KL penalty
    ref_model = None
    if args.kl_beta > 0:
        print(f"Loading reference model: {args.model_name}", flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        ).to(device)
        ref_model.resize_token_embeddings(len(tokenizer))
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # LoRA
    if args.use_lora:
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            modules_to_save=['embed_tokens', 'lm_head'],
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    params_total = sum(p.numel() for p in model.parameters())
    params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params_total: {params_total}  params_learnable: {params_learnable}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # ---- Training loop ----------------------------------------------------
    num_steps = 0
    losses = []

    tbar = tqdm(
        range(0, len(data_pairs), args.batch_size),
        desc="GRPO Training",
    )

    for batch_start in tbar:
        batch_items = data_pairs[batch_start: batch_start + args.batch_size]

        if not batch_items:
            continue

        loss, avg_reward = grpo_step(
            model, ref_model, tokenizer, batch_items, device, args,
            step_token_names,
        )

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        loss_val = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else 0.0
        tbar.set_description(
            f"Loss: {loss_val:.4f} | Reward: {avg_reward:.3f}"
        )

        losses.append({
            "step": num_steps,
            "loss": loss_val,
            "reward": avg_reward,
        })

        num_steps += 1
        if num_steps % args.save_interval == 0:
            save_config = train_config.copy()
            save_config["num_steps"] = num_steps
            model_save_dir = os.path.join(
                train_config["model_save_dir"], str(num_steps)
            )
            save_run(save_config, model_save_dir, model, tokenizer, None)

    # Save final losses
    if os.path.exists(train_config["model_save_dir"]):
        json_path = os.path.join(train_config["model_save_dir"], "losses.json")
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training for causal step ordering with Qwen3"
    )

    # Model & data
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", default="./data/recipenlg/recipenlg_clean.json")
    parser.add_argument("--bf16", default=1, type=int)

    # Data
    parser.add_argument("--num_samples", default=100_000, type=int)
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Number of prompts per GRPO step")
    parser.add_argument("--min_recipe_steps", default=3, type=int)
    parser.add_argument("--stp_max_steps", default=15, type=int)

    # GRPO hyperparameters
    parser.add_argument("--num_generations", default=4, type=int,
                        help="G: completions sampled per prompt")
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--max_prompt_length", default=512, type=int)
    parser.add_argument("--temperature", default=0.6, type=float,
                        help="Qwen3 thinking mode recommended: 0.6")
    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Qwen3 thinking mode recommended: 0.95")
    parser.add_argument("--top_k", default=20, type=int,
                        help="Qwen3 thinking mode recommended: 20")
    parser.add_argument("--clip_epsilon", default=0.2, type=float,
                        help="PPO-style clipping range")
    parser.add_argument("--kl_beta", default=0.01, type=float,
                        help="KL penalty coefficient against reference model")

    # Training
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--save_interval", default=500, type=int)

    # LoRA
    parser.add_argument("--use_lora", default=1, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)

    # Compat keys for setup_config directory naming
    parser.add_argument("--batch_mode", default="grpo", type=str)
    parser.add_argument("--prompt_type", default="grpo_step_tokens", type=str)
    parser.add_argument("--attn_mask_type", default="full", type=str)
    parser.add_argument("--loss_mask_type", default="completion_only", type=str)
    parser.add_argument("--use_clm", default=0, type=int)
    parser.add_argument("--use_kl", default=0, type=int)
    parser.add_argument("--use_mml", default=0, type=int)
    parser.add_argument("--use_grl", default=0, type=int)
    parser.add_argument("--use_stp", default=1, type=int)
    parser.add_argument("--use_cos", default=0, type=int)
    parser.add_argument("--init_from_eos", default=0, type=int)
    parser.add_argument("--use_abs_pe", default=0, type=int)
    parser.add_argument("--activations", default="real", type=str)
    parser.add_argument("--neg_ratio", default=0.0, type=float)

    args = parser.parse_args()
    main(args)
