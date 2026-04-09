"""
GRPO training for causal step ordering with Qwen3 models.

Given shuffled recipe steps with step tokens (pi_shuf), the model learns to:
1. Reason about step dependencies inside <think>...</think>
2. Output step tokens in the correct chronological order

Uses Group Relative Policy Optimization (GRPO) with verifiable rewards.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from peft import get_peft_model, LoraConfig
from utils_sys import save_run, setup_config
from tqdm.auto import tqdm
import argparse
import json
import random
import os
import re
import gc
import tempfile
import shutil


def build_grpo_prompt(item, tokenizer):
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

    # Build the shuffled steps with step tokens
    steps_text = ""
    for j, step in enumerate(shuf):
        steps_text += f"{j} {step.strip()}\n"

    # Ground truth: for each position in orig, find which step token (shuf index) it had
    ground_truth_order = []
    for orig_step in orig:
        shuf_idx = shuf.index(orig_step)
        ground_truth_order.append(shuf_idx)

    system_msg = (
        "You are given recipe steps which can either be in the correct or shuffled order. "
        "Each source step is labeled with a step index "
        "(e.g., 0, 1, ..., N, etc.). "
        "Your task is to reason about the correct chronological "
        "order of the steps, then output the step tokens in the correct order. "
        "If the original order is correct, "
        "then simply output the target step indices in the same order 0, 1, ..., N; "
        "otherwise, you will have to reorder the indices in your answer. "
        "First think step-by-step inside <think>...</think> tags, "
        "then output ONLY the step indices "
        "in the correct order, separated by spaces. "
        # "Think in at most 2-3 sentences.\n\n"
        "Respond in JSON format:\n"
        '{"answer_indices": [i, j, ..., k]}'
    )

    user_msg = f"{steps_text}\nOutput the step tokens in the correct chronological order."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt_text = tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=True,
                                                )
    return prompt_text, ground_truth_order


def parse_step_tokens_from_response(response_text):
    """
    Parse step token indices from model response.
    Tries JSON parsing first, falls back to regex extraction.

    Returns:
        list[int] or None if parsing fails
    """
    # Try to find content after </think>
    think_end = response_text.find("</think>")
    if think_end != -1:
        answer_part = response_text[think_end + len("</think>"):]
    else:
        answer_part = response_text

    # Clean up
    answer_part = answer_part.replace("<|im_end|>", "").strip()

    # Try JSON parsing
    try:
        parsed = json.loads(answer_part)
        indices = parsed.get("answer_indices", None)
        if indices is not None and isinstance(indices, list):
            return [int(i) for i in indices]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: try to find JSON in the text
    match = re.search(r'\{[^}]*"answer_indices"\s*:\s*\[([^\]]*)\][^}]*\}', answer_part)
    if match:
        try:
            nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
            return nums
        except ValueError:
            pass

    # Fallback: extract all integers
    matches = re.findall(r"(\d+)", answer_part)
    if not matches:
        return None

    return [int(m) for m in matches]


def compute_reward(predicted_order, ground_truth_order, n_steps, response_text=""):
    """
    Compute binary reward for a predicted step ordering.
    Returns 1.0 if the prediction exactly matches ground truth, 0.0 otherwise.
    """
    if predicted_order is None:
        return 0.0
    if predicted_order == ground_truth_order:
        return 1.0
    return 0.0


def compute_log_probs(model, input_ids, attention_mask, response_start_idx):
    """Compute per-token log probs for the response portion."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    response_mask = torch.zeros_like(token_log_probs)
    response_mask[:, response_start_idx - 1:] = 1.0
    return token_log_probs, response_mask


def init_vllm(model_name, args):
    """Initialize vLLM engine on a separate GPU for fast generation."""
    from vllm import LLM, SamplingParams
    vllm_engine = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        enable_chunked_prefill=True,
    )
    return vllm_engine


def sync_weights_to_vllm(model, vllm_engine, tokenizer, tmpdir, args):
    """Save training model weights and reload into vLLM engine on GPU 1."""
    # Save the current training model weights to a temp directory
    save_path = os.path.join(tmpdir, "vllm_sync")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # Reload the vLLM model from the saved weights
    del vllm_engine
    gc.collect()
    torch.cuda.empty_cache()
    # Pin to GPU 1 for the new vLLM workers
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from vllm import LLM
    vllm_engine = LLM(
        model=save_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        enable_chunked_prefill=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    return vllm_engine


def vllm_generate_completions(vllm_engine, tokenizer, prompts_data, args):
    """
    Use vLLM to generate G completions per prompt (batched, much faster).
    Returns list of (prompt_text, ground_truth_order, n_steps, list_of_response_texts, prompt_ids).
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=args.num_generations,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Build all prompts
    prompt_infos = []
    prompt_texts = []
    for item in prompts_data:
        prompt_text, ground_truth_order = build_grpo_prompt(item, tokenizer)
        n_steps = len(item['shuf'])
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_len = prompt_enc['input_ids'].shape[1]
        if prompt_len > args.max_prompt_length:
            continue
        prompt_infos.append((prompt_text, ground_truth_order, n_steps, prompt_enc['input_ids']))
        prompt_texts.append(prompt_text)

    if not prompt_texts:
        return []

    # Batch generate all prompts x G completions at once
    outputs = vllm_engine.generate(prompt_texts, sampling_params)

    results = []
    for info, output in zip(prompt_infos, outputs):
        prompt_text, ground_truth_order, n_steps, prompt_ids = info
        response_texts = [o.text for o in output.outputs]
        results.append((prompt_text, ground_truth_order, n_steps, response_texts, prompt_ids))

    return results


def grpo_generate(model, tokenizer, streamer, prompts_data, device, args, vllm_engine=None):
    """
    Generate G completions per prompt, compute rewards and old log-probs.
    Returns data needed for multiple policy gradient epochs.
    """
    G = args.num_generations
    all_prompt_data = []
    total_reward = 0.0
    n_prompts = 0

    if vllm_engine is not None:
        # ---- vLLM fast path: batched generation on separate GPU ----
        vllm_results = vllm_generate_completions(vllm_engine, tokenizer, prompts_data, args)

        for prompt_text, ground_truth_order, n_steps, response_texts, prompt_ids_cpu in vllm_results:
            prompt_ids = prompt_ids_cpu.to(device)
            prompt_len = prompt_ids.shape[1]

            # Tokenize each completion into full sequences (prompt + response)
            completions_ids = []
            rewards = []
            for g, resp_text in enumerate(response_texts):
                full_text = prompt_text + resp_text
                full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
                full_ids = full_enc['input_ids'].to(device)
                completions_ids.append(full_ids)

                predicted_order = parse_step_tokens_from_response(resp_text)
                reward = compute_reward(predicted_order, ground_truth_order, n_steps, resp_text)
                rewards.append(reward)
                print(f"\n[vLLM Gen {g+1}/{G}] reward={reward:.3f}\n{resp_text[:200]}...", flush=True)

            if not rewards:
                continue

            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            total_reward += rewards_t.mean().item()
            n_prompts += 1

            advantages = rewards_t - 0.5  # fixed baseline for binary rewards

            old_log_probs_list = []
            model.eval()
            with torch.no_grad():
                for g in range(len(completions_ids)):
                    full_ids = completions_ids[g].to(device)
                    attn = torch.ones_like(full_ids)
                    lp, rm = compute_log_probs(model, full_ids, attn, prompt_len)
                    old_log_probs_list.append(lp)

            all_prompt_data.append({
                "completions_ids": completions_ids,
                "advantages": advantages,
                "old_log_probs": old_log_probs_list,
                "prompt_len": prompt_len,
            })

    else:
        # ---- Original HF generate path ----
        for item in prompts_data:
            prompt_text, ground_truth_order = build_grpo_prompt(item, tokenizer)
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
                attn = torch.ones_like(prompt_ids)
                print(f"\n--- Generating {G} completions ---", flush=True)
                for g in range(G):
                    output_ids = model.generate(
                        prompt_ids,
                        attention_mask=attn,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        pad_token_id=tokenizer.pad_token_id,
                        streamer=streamer,
                    )
                    response_ids = output_ids[0, prompt_len:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                    completions_ids.append(output_ids)
                    predicted_order = parse_step_tokens_from_response(response_text)
                    reward = compute_reward(predicted_order, ground_truth_order, n_steps, response_text)
                    rewards.append(reward)
                    # print(f"\n[Gen {g+1}/{G}] reward={reward:.3f}\n{response_text}", flush=True)

            if not rewards:
                continue

            torch.cuda.empty_cache()
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            total_reward += rewards_t.mean().item()
            n_prompts += 1

            # ---- Fixed baseline advantages ----------------------------------------
            advantages = rewards_t - 0.5  # fixed baseline for binary rewards

            # ---- Old log probs (frozen at sampling time) -------------------------
            old_log_probs_list = []
            model.eval()
            with torch.no_grad():
                for g in range(G):
                    full_ids = completions_ids[g].to(device)
                    attn = torch.ones_like(full_ids)
                    lp, rm = compute_log_probs(model, full_ids, attn, prompt_len)
                    old_log_probs_list.append(lp)

            all_prompt_data.append({
                "completions_ids": completions_ids,
                "advantages": advantages,
                "old_log_probs": old_log_probs_list,
                "prompt_len": prompt_len,
            })

    avg_reward = total_reward / n_prompts if n_prompts > 0 else 0.0
    return all_prompt_data, avg_reward


def grpo_policy_loss(model, ref_model, all_prompt_data, device, args):
    """
    Compute clipped policy gradient loss over pre-generated completions.
    Called once per PPO epoch.
    """
    G = args.num_generations
    total_loss = torch.tensor(0.0, device=device)

    if not all_prompt_data:
        return total_loss

    model.train()
    for pdata in all_prompt_data:
        completions_ids = pdata["completions_ids"]
        advantages = pdata["advantages"]
        old_log_probs = pdata["old_log_probs"]
        prompt_len = pdata["prompt_len"]

        prompt_loss = torch.tensor(0.0, device=device)
        for g in range(G):
            full_ids = completions_ids[g].to(device)
            attn = torch.ones_like(full_ids)

            new_lp, resp_mask = compute_log_probs(
                model, full_ids, attn, prompt_len
            )
            old_lp = old_log_probs[g].detach()

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

                kl_per_token = new_lp - ref_token_lp
                kl_loss = (kl_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)

            prompt_loss = prompt_loss + masked_loss + args.kl_beta * kl_loss

        prompt_loss = prompt_loss / G
        total_loss = total_loss + prompt_loss

    total_loss = total_loss / len(all_prompt_data)
    return total_loss


def make_shuffled_dataset(data, min_steps=3, max_steps=15, neg_ratio=0.0):
    """
    Create a dataset of (orig, shuf) pairs.
    neg_ratio controls the fraction of samples where shuf == orig (correct order).
    """
    step_list_orig = [
        x['directions'] for x in data
        if len(set(x['directions'])) == len(x['directions'])
        and len(x['directions']) > 1
        and min_steps <= len(x['directions']) <= max_steps
    ]
    print(f"Filtered dataset: {len(step_list_orig)} recipes "
          f"({min_steps}–{max_steps} steps)")

    pairs = []
    for orig in step_list_orig:
        n = len(orig)
        if random.random() < neg_ratio:
            # Keep original order
            pairs.append({'orig': orig, 'shuf': list(orig), 'binary_label': 1})
        else:
            shuf = random.sample(orig, n)
            attempts = 0
            while shuf == orig and attempts < 100:
                shuf = random.sample(orig, n)
                attempts += 1
            if shuf != orig:
                pairs.append({'orig': orig, 'shuf': shuf, 'binary_label': 0})
    return pairs


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
        neg_ratio=args.neg_ratio,
    )
    random.shuffle(data_pairs)
    print(f"Training pairs: {len(data_pairs)}")

    # Use cuda:0 for training; cuda:1 reserved for vLLM if enabled
    if args.use_vllm and torch.cuda.device_count() >= 2:
        device = "cuda:0"
        os.environ["CUDA_VISIBLE_DEVICES_VLLM"] = "1"  # informational
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Model & tokenizer ------------------------------------------------
    print(f"Loading model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    streamer = TextStreamer(tokenizer)
    # step_token_names = [str(i) for i in range(args.stp_max_steps)]
    # if args.add_step_tokens:
    #     # Add step tokens to vocabulary
    #     tokenizer.add_tokens(step_token_names, special_tokens=True)
    #     model.resize_token_embeddings(len(tokenizer))

    #     step_token_id_map = {
    #         i: tokenizer.convert_tokens_to_ids(f"S_{i}")
    #         for i in range(args.stp_max_steps)
    #     }

    # Reference model for KL penalty
    ref_model = None
    if args.kl_beta > 0:
        print(f"Loading reference model: {args.model_name}", flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        ref_model.resize_token_embeddings(len(tokenizer))
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # LoRA
    if args.use_lora:
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
                'embed_tokens',
                'lm_head',
                ],
            # modules_to_save=['embed_tokens', 'lm_head'],
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

    params_total = sum(p.numel() for p in model.parameters())
    params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params_total: {params_total}  params_learnable: {params_learnable}")

    # ---- vLLM engine for fast generation -----------------------------------
    vllm_engine = None
    vllm_tmpdir = None
    if args.use_vllm:
        print("Initializing vLLM engine on cuda:1...", flush=True)
        # Pin vLLM workers to GPU 1 only. The main process already has
        # PyTorch initialized on cuda:0, so changing CUDA_VISIBLE_DEVICES
        # here only affects newly spawned vLLM worker processes.
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        vllm_engine = init_vllm(args.model_name, args)
        # Restore so the main process can still see both GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        vllm_tmpdir = tempfile.mkdtemp(prefix="grpo_vllm_sync_")
        print("vLLM engine ready.", flush=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    # LR scheduler: linear warmup + cosine decay
    total_steps = len(range(0, len(data_pairs), args.batch_size))
    num_update_steps = total_steps // args.grad_accum_steps
    warmup_steps = int(num_update_steps * args.warmup_ratio)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps - warmup_steps, eta_min=args.lr * 0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(num_update_steps, 1), eta_min=args.lr * 0.1)
    print(f"Total steps: {total_steps}, update steps: {num_update_steps}, warmup steps: {warmup_steps}", flush=True)

    # ---- Training loop ----------------------------------------------------
    optimizer.zero_grad()
    num_steps = 0
    losses = []

    tbar = tqdm(
        range(0, len(data_pairs), args.batch_size),
        desc="GRPO Training",
    )

    # import pdb; pdb.set_trace()

    for batch_start in tbar:
        batch_items = data_pairs[batch_start: batch_start + args.batch_size]

        if not batch_items:
            continue

        # Generate completions and compute rewards + old log-probs (once)
        all_prompt_data, avg_reward = grpo_generate(model, tokenizer, streamer, batch_items, device, args, vllm_engine=vllm_engine)

        # Single backward pass, accumulate gradients across batches
        loss = grpo_policy_loss(model, ref_model, all_prompt_data, device, args)

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            if (num_steps + 1) % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Sync updated weights to vLLM engine
                if vllm_engine is not None:
                    print("Syncing weights to vLLM...", flush=True)
                    vllm_engine = sync_weights_to_vllm(model, vllm_engine, tokenizer, vllm_tmpdir, args)

        loss_val = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else 0.0
        updated = (num_steps + 1) % args.grad_accum_steps == 0 and isinstance(loss, torch.Tensor) and loss.requires_grad
        print(f"\n[Step {num_steps}] Loss: {loss_val:.4f} | Reward: {avg_reward:.3f} | Updated: {updated}", flush=True)
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

    # Save final model
    save_config = train_config.copy()
    save_config["num_steps"] = num_steps
    model_save_dir = os.path.join(
        train_config["model_save_dir"], "final"
    )
    save_run(save_config, model_save_dir, model, tokenizer, None)

    # Save final losses
    json_path = os.path.join(train_config["model_save_dir"], "losses.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(losses, f, ensure_ascii=False, indent=4)

    # Cleanup vLLM temp dir
    if vllm_tmpdir and os.path.exists(vllm_tmpdir):
        shutil.rmtree(vllm_tmpdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training for causal step ordering with Qwen3"
    )

    parser.add_argument("--config", default=None, type=str,
                        help="Path to JSON config file that overrides arguments")

    # Model & data
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--data_path", default="./data/recipenlg/recipenlg_clean.json")
    parser.add_argument("--bf16", default=1, type=int)

    # Data
    parser.add_argument("--num_samples", default=100_000, type=int)
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Number of prompts per GRPO step")
    parser.add_argument("--min_recipe_steps", default=3, type=int)
    parser.add_argument("--stp_max_steps", default=15, type=int)

    # GRPO hyperparameters
    parser.add_argument("--num_generations", default=8, type=int,
                        help="G: completions sampled per prompt")
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--max_prompt_length", default=512, type=int)
    parser.add_argument("--temperature", default=0.6, type=float, help="Qwen3 thinking mode recommended: 0.6")
    parser.add_argument("--top_p", default=0.95, type=float, help="Qwen3 thinking mode recommended: 0.95")
    parser.add_argument("--top_k", default=20, type=int, help="Qwen3 thinking mode recommended: 20")
    parser.add_argument("--clip_epsilon", default=0.2, type=float, help="PPO-style clipping range")
    parser.add_argument("--kl_beta", default=0.05, type=float, help="KL penalty coefficient against reference model")

    # Training
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--grad_accum_steps", default=4, type=int)
    parser.add_argument("--save_interval", default=100, type=int)

    # LoRA
    parser.add_argument("--use_lora", default=1, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)

    # vLLM
    parser.add_argument("--use_vllm", default=0, type=int,
                        help="Use vLLM for fast generation on a separate GPU")

    # Compat keys for setup_config directory naming
    parser.add_argument("--batch_mode", default="grpo", type=str)
    parser.add_argument("--prompt_type", default="grpo_step_tokens", type=str)
    parser.add_argument("--attn_mask_type", default="full", type=str)
    parser.add_argument("--clm_mask_type", default="completion_only", type=str)
    parser.add_argument("--use_clm", default=0, type=int)
    parser.add_argument("--use_kl", default=0, type=int)
    parser.add_argument("--use_mml", default=0, type=int)
    parser.add_argument("--use_grl", default=0, type=int)
    parser.add_argument("--use_stp", default=1, type=int)
    parser.add_argument("--add_step_tokens", default=1, type=int)
    parser.add_argument("--use_cos", default=0, type=int)
    parser.add_argument("--init_from_eos", default=0, type=int)
    parser.add_argument("--use_abs_pe", default=0, type=int)
    parser.add_argument("--activations", default="real", type=str)
    parser.add_argument("--neg_ratio", default=0.5, type=float)
    
    parser.add_argument("--revision", default=None, type=str,
                        help="Model revision/checkpoint to load (e.g. 'step4000' for Pythia early checkpoints)")

    args = parser.parse_args()

    # Override args with config file if provided
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        print(f"Loaded config from {args.config}", flush=True)

    main(args)
