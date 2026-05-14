"""
CaT-Bench ICL evaluation with model.generate.

Prompts a model with recipe steps + binary question, generates a yes/no
answer (optionally with thinking), and evaluates based on the generated
answer and its token probability.

Usage:
  python src/cat_bench_icl.py --model_dir Qwen/Qwen3-1.7B
  python src/cat_bench_icl.py --model_dir Qwen/Qwen3-1.7B --thinking 0
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftConfig
from tqdm.auto import tqdm
import pandas as pd
import os
import json
import re
import argparse
import numpy as np
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def format_question(row):
    """Format a single CaT-Bench sample into user question text."""
    steps = list(row["steps"])
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return (
        f"Recipe: {row['title']}\n\n"
        f"Steps:\n{steps_text}\n\n"
        f"{row['binary_question']}\n\n"
        f'Answer with just "yes" or "no".'
    )


def build_prompt(row, tokenizer, thinking=True, icl_examples=None):
    """Build a chat prompt with ICL examples followed by the test question."""
    messages = []

    # Add ICL examples as prior user/assistant turn pairs
    if icl_examples is not None:
        for _, ex in icl_examples.iterrows():
            messages.append({"role": "user", "content": format_question(ex)})
            answer = "yes" if ex["label"] == 1 else "no"
            messages.append({"role": "assistant", "content": answer})

    # Add the test question
    messages.append({"role": "user", "content": format_question(row)})

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return prompt_text


def parse_answer(text):
    """Extract yes/no from generated text, handling <think> blocks."""
    # Strip thinking block if present
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):]

    text = text.strip().lower()
    # Try JSON format first
    match = re.search(r'"answer"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Direct yes/no
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    # Search anywhere
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def sample_icl_examples(df_train, n_icl, sample_type="real"):
    """Sample balanced ICL examples from train set."""
    df = df_train[df_train["type"] == sample_type] if sample_type != "all" else df_train
    # Sample n_icl per (label, direction) group for balance
    icl = df.groupby(
        ["label", "direction"], group_keys=False
    ).apply(lambda x: x.sample(n=min(n_icl, len(x)), replace=False))
    return icl.sample(frac=1)  # shuffle


def evaluate_batch(batch_df, tokenizer, model, device, thinking=True,
                   max_new_tokens=1024, yes_ids=None, no_ids=None, streamer=None,
                   df_train=None, n_icl=3):
    """Generate answers for a batch and return per-sample results."""
    batch_df = batch_df.reset_index(drop=True)

    # Sample ICL examples once per batch
    icl_examples = None
    if df_train is not None and n_icl > 0:
        icl_examples = sample_icl_examples(df_train, n_icl)

    prompts = []
    for _, row in batch_df.iterrows():
        prompts.append(build_prompt(row, tokenizer, thinking=thinking,
                                    icl_examples=icl_examples))

    # Tokenize with left padding for batch generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        streamer=streamer,
    )
    if thinking:
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    with torch.no_grad():
        gen_outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    tokenizer.padding_side = "right"

    results = []
    for i in range(len(batch_df)):
        row = batch_df.iloc[i]
        gen_ids = gen_outputs.sequences[i, prompt_len:]

        # Strip EOS
        if tokenizer.eos_token_id is not None:
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0]]

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        answer = parse_answer(gen_text)

        # Find the yes/no token position in generated sequence and get its prob
        # Look for the first yes/no token after </think> (or from start if no thinking)

        # Find </think> token id
        think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        last_think_end_id = think_end_ids[-1]
        scores_offset = 0
        if thinking and last_think_end_id in gen_ids:
            scores_offset = torch.where(gen_ids == last_think_end_id)[0][0].item()
            gen_ids = gen_ids[scores_offset:]

        answer_token_idx = None
        for t in range(len(gen_ids)):
            tok_id = gen_ids[t].item()
            if tok_id in yes_ids or tok_id in no_ids:
                answer_token_idx = scores_offset + t
                break

        prob_yes = 0.0
        prob_no = 0.0
        if answer_token_idx is not None and answer_token_idx < len(gen_outputs.scores):
            logits = gen_outputs.scores[answer_token_idx][i]
            probs = F.softmax(logits, dim=-1)
            prob_yes = probs[yes_ids].sum().item()
            prob_no = probs[no_ids].sum().item()
        # Map answer to prediction
        label_map = {"yes": 1, "no": 0}
        pred_label = label_map.get(answer, -1)
        gold_label = int(row["label"])

        results.append({
            "gold_label": gold_label,
            "pred_label": pred_label,
            "answer": answer,
            "generated_text": gen_text,
            "prob_yes": prob_yes,
            "prob_no": prob_no,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    train_path = "./data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json"
    test_path = "./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json"

    df_train = pd.read_json(train_path)
    df_test = pd.read_json(test_path)
    if args.sample_type != "all":
        df_test = df_test[df_test["type"] == args.sample_type]
    if args.num_samples > 0:
        df_test = df_test.sample(frac=1)
        df_test = df_test.head(args.num_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Walk model_dir for checkpoints or treat as single model
    if not os.path.exists(args.model_dir):
        model_list = [{"path": args.model_dir, "num_steps": 0}]
    else:
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for filename in files:
                if filename == "train_config.json":
                    with open(os.path.join(root, filename), "r", encoding="utf8") as f:
                        num_steps = json.load(f)["num_steps"]
                    model_list.append({"path": root, "num_steps": num_steps})
        if not model_list:
            model_list = [{"path": args.model_dir, "num_steps": 0}]
        model_list = sorted(model_list, key=lambda x: x["num_steps"])

    for m in model_list:
        model_name = m["path"]

        print(f"Loading model from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        adapter_config_path = os.path.join(model_name, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            print("-> Detected LoRA adapter. Using PEFT two-stage loading...")
            config = PeftConfig.from_pretrained(model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, dtype=torch.bfloat16,
            ).to(device)

            from safetensors import safe_open
            adapter_safetensors = os.path.join(model_name, "adapter_model.safetensors")
            checkpoint_vocab_size = None
            if os.path.exists(adapter_safetensors):
                with safe_open(adapter_safetensors, framework="pt") as f:
                    for key in f.keys():
                        if "modules_to_save" in key and ("embed_tokens" in key or "lm_head" in key):
                            checkpoint_vocab_size = f.get_slice(key).get_shape()[0]
                            break
            if checkpoint_vocab_size is not None and checkpoint_vocab_size != base_model.config.vocab_size:
                base_model.resize_token_embeddings(checkpoint_vocab_size)

            model = PeftModel.from_pretrained(base_model, model_name)
        else:
            print(f"-> Loading model directly...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16,
            ).to(device)
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))

        model.eval()
        streamer = TextStreamer(tokenizer, skip_special_tokens=False) if args.verbose else None

        # Yes / No token IDs
        yes_variants = [" yes", " Yes", "yes", "Yes"]
        no_variants = [" no", " No", "no", "No"]

        def get_valid_ids(words):
            ids = []
            for w in words:
                token_ids = tokenizer.encode(w, add_special_tokens=False)
                if len(token_ids) == 1:
                    ids.append(token_ids[0])
            return list(set(ids))

        yes_ids = get_valid_ids(yes_variants)
        no_ids = get_valid_ids(no_variants)
        print("Yes tokens:", tokenizer.convert_ids_to_tokens(yes_ids))
        print("No tokens :", tokenizer.convert_ids_to_tokens(no_ids))

        # Evaluate
        all_results = []
        for i in tqdm(range(0, len(df_test), args.batch_size)):
            batch = df_test.iloc[i : i + args.batch_size]
            batch_results = evaluate_batch(
                batch, tokenizer, model, device,
                thinking=args.thinking,
                max_new_tokens=args.max_new_tokens,
                yes_ids=yes_ids, no_ids=no_ids,
                streamer=streamer,
                df_train=df_train, n_icl=args.n_icl,
            )
            all_results.extend(batch_results)

            # Save decoded first sample for inspection
            if i == 0:
                os.makedirs("./misc", exist_ok=True)
                with open("./misc/cat_bench_icl_sample.txt", "w", encoding="utf-8") as f:
                    f.write(batch_results[0]["generated_text"])
                print("Saved decoded first sample to ./misc/cat_bench_icl_sample.txt")

        # Metrics
        gold = [r["gold_label"] for r in all_results]
        pred = [r["pred_label"] for r in all_results]

        # Handle invalid predictions for metrics
        n_invalid = sum(1 for p in pred if p == -1)
        # For metrics, map invalid to opposite of gold
        pred_safe = []
        for g, p in zip(gold, pred):
            pred_safe.append(p if p != -1 else (1 - g))

        scores = [r["prob_yes"] - r["prob_no"] for r in all_results]

        acc = accuracy_score(gold, pred_safe)
        f1 = f1_score(gold, pred_safe, average="macro")

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1:.4f}")
        print(f"Invalid predictions: {n_invalid}/{len(pred)}")
        if len(set(gold)) == 2 and any(s != 0.0 for s in scores):
            roc = roc_auc_score(gold, scores)
            pr = average_precision_score(gold, scores)
            print(f"ROC AUC: {roc:.4f}")
            print(f"PR AUC: {pr:.4f}")
        else:
            roc = None
            pr = None
        print(classification_report(gold, pred_safe, digits=4))

        # Save
        task_name = "cat_bench_icl" if args.thinking else "cat_bench_icl_baseline"
        model_leaf = os.path.basename(os.path.normpath(model_name))
        save_dir = os.path.join(args.results_base_dir, task_name, model_leaf)
        os.makedirs(save_dir, exist_ok=True)

        out = {
            "eval_config": vars(args),
            "results": {
                "accuracy": acc,
                "f1_macro": f1,
                "roc_auc": roc,
                "pr_auc": pr,
                "n_invalid": n_invalid,
                "n_total": len(gold),
            },
            "predictions": all_results,
        }
        json_path = os.path.join(save_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
        print(f"Saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaT-Bench ICL evaluation with model.generate")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sample_type", default="real", choices=["real", "all"])
    parser.add_argument("--num_samples", default=0, type=int, help="0 = all")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--thinking", default=1, type=int,
                        help="1 = thinking mode; 0 = no thinking")
    parser.add_argument("--n_icl", default=0, type=int,
                        help="Number of ICL examples per (label, direction) group")
    parser.add_argument("--results_base_dir", default="./results", type=str)
    parser.add_argument("--verbose", default=0, type=int, help="Stream generated tokens to stdout")
    args = parser.parse_args()
    main(args)
