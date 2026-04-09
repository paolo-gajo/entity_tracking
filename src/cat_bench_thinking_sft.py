"""
SFT Qwen3 on CATBench thinking data.

Trains the model to produce GPT-4o explanations inside <think>...</think> tokens
and a binary yes/no answer after </think>.

Usage:
  python src/cat_bench_thinking_sft.py \
    --model_name Qwen/Qwen3-0.6B \
    --batch_size 4 \
    --max_train_steps 10000 \
    --use_lora 0 \
    --train 1 \
    --eval 0 \
    --save_interval 1000
"""

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import argparse
import json
import os
import re
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score


def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThinkingSFTDataset(Dataset):
    """Each sample is (input_ids, labels) where labels are masked on the prompt."""

    def __init__(self, data, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            prompt_text, full_text = self._format(item)
            self.samples.append((prompt_text, full_text))

    def _format(self, item):
        """Build prompt (user turn) and full sequence (user + assistant with thinking)."""
        user_content = item["model_input"]
        thinking = item["why_answer"]
        answer = item["binary_answer"]

        format_instruction = '\n\nProvide my_answer = "yes|no" in a json {"answer": "my_answer"}.'
        user_content = user_content + format_instruction
        assistant_content = f'<think>\n{thinking}\n</think>\n\n{{"answer": "{answer}"}}'

        # Prompt: user turn + generation prompt (via chat template)
        prompt_msgs = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        ) + "<think>\n"

        # Full: user turn + assistant response (via chat template)
        full_msgs = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        return prompt_text, full_text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_text, full_text = self.samples[idx]

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate to max_length
        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        # Labels: -100 for prompt tokens, actual ids for assistant tokens
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def save_thinking_prompt(batch, tokenizer, path="./misc/thinking_prompt.txt"):
    """Save decoded first batch to file for inspection: all tokens and trained-on tokens."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prompt = ""
    for i in range(batch["input_ids"].shape[0]):
        decoded_all = tokenizer.decode(batch["input_ids"][i])
        attn_mask = batch["attention_mask"][i] == 1
        attn_tokens = batch["input_ids"][i][attn_mask]
        attn_tokens_decoded = tokenizer.decode(attn_tokens)
        # Trained-on tokens: where labels != -100
        train_mask = batch["labels"][i] != -100
        train_tokens = batch["input_ids"][i][train_mask]
        train_tokens_decoded = tokenizer.decode(train_tokens)
        prompt += (
            f"{'#' * 100}\n\n"
            f"ALL input_ids:\n`{decoded_all}`\n\n"
            f"{'-' * 100}\n\n"
            f"ATTN TOKENS:\n`{attn_tokens_decoded}`\n\n"
            f"{'-' * 100}\n\n"
            f"TRAINED-ON TOKENS (labels != -100):\n`{train_tokens_decoded}`\n\n"
            f"{'#' * 100}\n\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Saved thinking prompt to {path}", flush=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, data, device, max_new_tokens=256, batch_size=4, verbose=False):
    """Generate answers and compute accuracy."""
    model.eval()
    preds = []
    correct = 0
    total = 0

    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch_items = data[i : i + batch_size]
        prompts = []
        gold_answers = []
        for item in batch_items:
            format_instruction = '\n\nProvide my_answer = "yes|no" in a json {"answer": "my_answer"}.'
            user_content = item["model_input"] + format_instruction
            prompt_msgs = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            ) + "<think>\n"
            prompts.append(prompt)
            gold_answers.append(item["binary_answer"].strip().lower())

        # Tokenize with left padding for batch generation
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        streamer = TextStreamer(tokenizer, skip_special_tokens=False) if verbose else None
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                streamer=streamer,
            )

        # Decode only the generated part
        for j, output_ids in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Parse thinking block and answer
            thinking_block, answer = parse_generation(generated_text)
            pred_answer = answer.strip().lower() if answer else ""

            is_correct = pred_answer == gold_answers[j]
            correct += int(is_correct)
            total += 1

            preds.append({
                "thinking_block": thinking_block,
                "answer": pred_answer,
                "gold_answer": gold_answers[j],
                "correct": is_correct,
                "question_type": batch_items[j].get("question_type", ""),
            })

    tokenizer.padding_side = "right"

    accuracy = correct / total if total > 0 else 0.0

    # Compute F1 (binary: "yes"=1, "no"=0; invalid predictions mapped to -1 then excluded)
    label_map = {"yes": 1, "no": 0}
    gold_labels = [label_map.get(p["gold_answer"], -1) for p in preds]
    pred_labels = [label_map.get(p["answer"], -1) for p in preds]

    # For overall F1, treat invalid predictions as wrong (map to opposite of gold)
    pred_labels_safe = []
    for g, p in zip(gold_labels, pred_labels):
        pred_labels_safe.append(p if p != -1 else (1 - g if g != -1 else 0))

    f1_macro = f1_score(gold_labels, pred_labels_safe, average="macro", zero_division=0)
    f1_binary_yes = f1_score(gold_labels, pred_labels_safe, pos_label=1, average="binary", zero_division=0)
    f1_binary_no = f1_score(gold_labels, pred_labels_safe, pos_label=0, average="binary", zero_division=0)
    precision_macro = precision_score(gold_labels, pred_labels_safe, average="macro", zero_division=0)
    recall_macro = recall_score(gold_labels, pred_labels_safe, average="macro", zero_division=0)

    n_invalid = sum(1 for p in pred_labels if p == -1)

    # Per question_type accuracy and F1
    type_stats = {}
    for p in preds:
        qt = p["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"correct": 0, "total": 0, "gold": [], "pred": []}
        type_stats[qt]["total"] += 1
        type_stats[qt]["correct"] += int(p["correct"])
        type_stats[qt]["gold"].append(label_map.get(p["gold_answer"], -1))
        pl = label_map.get(p["answer"], -1)
        g = label_map.get(p["gold_answer"], -1)
        type_stats[qt]["pred"].append(pl if pl != -1 else (1 - g if g != -1 else 0))
    for qt in type_stats:
        s = type_stats[qt]
        s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        s["f1_macro"] = f1_score(s["gold"], s["pred"], average="macro", zero_division=0)
        del s["gold"], s["pred"]

    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_yes": f1_binary_yes,
        "f1_no": f1_binary_no,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "n_invalid_preds": n_invalid,
        "correct": correct,
        "total": total,
        "per_type": type_stats,
    }
    return results, preds


def parse_generation(text):
    """Extract thinking block and final answer from generated text."""
    # Try to find </think> delimiter
    think_end = text.find("</think>")
    if think_end != -1:
        thinking_block = text[:think_end].strip()
        after_think = text[think_end + len("</think>"):]
        after_think = after_think.replace("<|im_end|>", "").strip()
    else:
        thinking_block = ""
        after_think = text.replace("<|im_end|>", "").strip()

    # Try to parse JSON answer
    answer = ""
    try:
        parsed = json.loads(after_think)
        answer = parsed.get("answer", "").strip().lower()
    except (json.JSONDecodeError, AttributeError):
        # Fallback: try to find JSON in the text
        match = re.search(r'\{[^}]*"answer"\s*:\s*"(yes|no)"[^}]*\}', after_think, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
        else:
            answer = after_think.strip().lower()
    return thinking_block, answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    time_string = get_current_time_string()
    model_short = args.model_name.replace("/", "_")
    model_save_dir = os.path.join("models", "cat_bench_thinking", model_short, time_string)
    results_dir = os.path.join("results", "cat_bench_thinking", model_short, time_string)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
    ).to(device)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    def load_split(split):
        path = os.path.join(args.data_dir, f"{args.data_prefix}_{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    if args.train:
        train_data = load_split("train")
        print(f"Train samples: {len(train_data)}", flush=True)

        val_data = load_split("val") if args.eval else None
        eval_every = max(1, int(args.max_train_steps * args.eval_interval))

        train_dataset = ThinkingSFTDataset(train_data, tokenizer, max_length=args.max_length)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
            drop_last=True,
        )

        optimizer = AdamW(model.parameters(), lr=args.lr)
        model.train()

        step = 0
        running_loss = 0.0
        data_iter = iter(train_loader)
        pbar = tqdm(total=args.max_train_steps, desc="Training")

        while step < args.max_train_steps:
            # Get next batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if step == 0:
                save_thinking_prompt(batch, tokenizer)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)

            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0

            # if step % args.save_interval == 0:
            #     save_path = os.path.join(model_save_dir, f"step_{step}")
            #     os.makedirs(save_path, exist_ok=True)
            #     if args.use_lora:
            #         model.save_pretrained(save_path)
            #     else:
            #         model.save_pretrained(save_path)
            #     tokenizer.save_pretrained(save_path)
            #     # Save config
            #     with open(os.path.join(save_path, "train_config.json"), "w") as f:
            #         json.dump({**vars(args), "step": step, "time_string": time_string}, f, indent=2)
            #     print(f"\nSaved checkpoint at step {step} to {save_path}", flush=True)

            if args.eval and step % eval_every == 0:
                print(f"\n[Step {step}] Running val evaluation...", flush=True)
                val_results, val_preds = evaluate(model, tokenizer, val_data, device,
                                                  max_new_tokens=args.max_new_tokens,
                                                  batch_size=args.eval_batch_size,
                                                  verbose=args.verbose)
                print(f"[Step {step}] Val acc: {val_results['accuracy']:.4f}  F1(macro): {val_results['f1_macro']:.4f}  F1(yes): {val_results['f1_yes']:.4f}  F1(no): {val_results['f1_no']:.4f}  invalid: {val_results['n_invalid_preds']}")
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, f"val_results_step{step}.json"), "w", encoding="utf-8") as f:
                    json.dump({"config": vars(args), "results": val_results, "preds": val_preds, "step": step}, f, indent=2, ensure_ascii=False)
                model.train()

        pbar.close()

        # Save final
        # final_path = os.path.join(model_save_dir, "final")
        os.makedirs(model_save_dir, exist_ok=True)
        if args.use_lora:
            model.save_pretrained(model_save_dir)
        else:
            model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
        with open(os.path.join(model_save_dir, "train_config.json"), "w") as f:
            json.dump({**vars(args), "step": step, "time_string": time_string}, f, indent=2)
        print(f"Saved final model to {model_save_dir}", flush=True)

    # -----------------------------------------------------------------------
    # Evaluation on test (always)
    # -----------------------------------------------------------------------
    test_data = load_split("test")
    if args.num_test_samples > 0:
        test_data = test_data[:args.num_test_samples]
    print(f"\nEvaluating on test ({len(test_data)} samples)...", flush=True)
    test_results, test_preds = evaluate(model, tokenizer, test_data, device,
                                        max_new_tokens=args.max_new_tokens,
                                        batch_size=args.eval_batch_size,
                                        verbose=args.verbose)
    print(f"Test accuracy: {test_results['accuracy']:.4f}  F1(macro): {test_results['f1_macro']:.4f}  F1(yes): {test_results['f1_yes']:.4f}  F1(no): {test_results['f1_no']:.4f}  invalid: {test_results['n_invalid_preds']}")
    for qt, s in test_results["per_type"].items():
        print(f"  {qt}: acc={s['accuracy']:.4f} f1={s['f1_macro']:.4f} ({s['correct']}/{s['total']})")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": test_results, "preds": test_preds}, f, indent=2, ensure_ascii=False)
    print(f"Saved test results to {results_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data_dir", type=str, default="data/cat_bench")
    parser.add_argument("--data_prefix", type=str, default="gpt-4o_explain_answer")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--eval", type=int, default=0)
    # parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=float, default=0.1,
                        help="Evaluate on val every this fraction of max_train_steps")
    parser.add_argument("--num_test_samples", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    args = parser.parse_args()
    main(args)
