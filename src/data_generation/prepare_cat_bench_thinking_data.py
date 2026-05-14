"""
Prepare train/val/test JSON files for CATBench thinking SFT.

Reads *_explain_binary.jsonl files from catplan-data-release generated_answers,
parses GPT-4o explanations, and creates splits with the Qwen3 chat format:

  user: {model_input}
  assistant: <think>{why_answer}</think>\n\n{binary_answer}

Usage:
  python src/prepare_cat_bench_thinking_data.py \
    --model_name gpt-4o-2024-05-13 \
    --train_ratio 0.8 --val_ratio 0.1
"""

import json
import os
import re
import random
import argparse
from glob import glob


def parse_model_answer(answer_str):
    """Extract why_answer and binary_answer from GPT-4o's markdown-wrapped JSON response."""
    # Strip markdown code fences if present
    cleaned = answer_str.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed["why_answer"], parsed["binary_answer"].strip().lower()
    except (json.JSONDecodeError, KeyError):
        return None, None


def strip_format_instruction(text):
    """Remove the trailing 'Format your answer as JSON ...' instruction from model_input."""
    # Match the format instruction at the end (with optional leading newlines)
    pattern = r'\n*Format your answer as JSON with the key value pairs.*$'
    return re.sub(pattern, "", text, flags=re.DOTALL).rstrip()


def load_jsonl_files(directory):
    """Load all *_explain_binary.jsonl files from a directory."""
    pattern = os.path.join(directory, "*_explain_binary.jsonl")
    files = sorted(glob(pattern))
    samples = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                why_answer, binary_answer = parse_model_answer(row["model_answer"])
                if why_answer is None:
                    print(f"WARNING: Could not parse model_answer in {fname}, plan_idx={row.get('plan_idx')}")
                    continue

                # Determine gold label from filename
                if fname.startswith("dependent"):
                    gold_label = "yes"
                elif fname.startswith("nondependent"):
                    gold_label = "no"
                else:
                    gold_label = binary_answer  # fallback

                # Only keep samples where GPT-4o's answer matches the gold label
                if binary_answer != gold_label:
                    continue

                # Strip the JSON format instruction from the prompt
                model_input = strip_format_instruction(row["model_input"])

                samples.append({
                    "model_input": model_input,
                    "why_answer": why_answer,
                    "binary_answer": gold_label,
                    "question_type": row.get("question_type", ""),
                    "plan_idx": row.get("plan_idx"),
                    "question_idx": row.get("question_idx"),
                    "title": row.get("title", ""),
                })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--data_root", type=str,
                        default="data/cat_bench/catplan-data-release/generated_answers")
    parser.add_argument("--output_dir", type=str, default="data/cat_bench")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    model_dir = os.path.join(args.data_root, args.model_name)

    # Check for train_must_why first, fall back to test_must_why
    train_dir = os.path.join(model_dir, "train_must_why")
    test_dir = os.path.join(model_dir, "test_must_why")

    if os.path.isdir(train_dir):
        print(f"Found train_must_why at {train_dir}")
        train_val_samples = load_jsonl_files(train_dir)
        test_samples = load_jsonl_files(test_dir)

        # Split train_val into train and val
        random.shuffle(train_val_samples)
        val_size = int(len(train_val_samples) * args.val_ratio / (args.train_ratio + args.val_ratio))
        val_samples = train_val_samples[:val_size]
        train_samples = train_val_samples[val_size:]
    else:
        print(f"No train_must_why found, using test_must_why for all splits")
        all_samples = load_jsonl_files(test_dir)

        # Split by plan_idx to avoid data leakage
        plan_ids = sorted(set(s["plan_idx"] for s in all_samples))
        random.shuffle(plan_ids)

        n = len(plan_ids)
        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)

        train_plans = set(plan_ids[:train_end])
        val_plans = set(plan_ids[train_end:val_end])
        test_plans = set(plan_ids[val_end:])

        train_samples = [s for s in all_samples if s["plan_idx"] in train_plans]
        val_samples = [s for s in all_samples if s["plan_idx"] in val_plans]
        test_samples = [s for s in all_samples if s["plan_idx"] in test_plans]

        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

    # Short model name for output files (e.g. "gpt-4o-2024-05-13" -> "gpt-4o")
    short_name = args.model_name.split("-2024")[0] if "-2024" in args.model_name else args.model_name

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        out_path = os.path.join(args.output_dir, f"{short_name}_explain_answer_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(split_data)} samples to {out_path}")


if __name__ == "__main__":
    main()
