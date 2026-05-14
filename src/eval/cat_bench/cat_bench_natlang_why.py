"""
Evaluate a trained model on CaT-Bench test_must_why questions in natural language.

For each question, the model is given the recipe steps and asked whether
one step must happen before/after another. The model reasons inside
<think>...</think> tags (Qwen3 native) then answers Yes or No.

Computes overall and per-question-type accuracy.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import re
import os
from tqdm import tqdm


# ======================================================================
# Prompt construction
# ======================================================================

def build_eval_prompt(sample, tokenizer):
    """
    Build a chat prompt for a CaT-Bench binary ordering question.
    """
    steps = sample['steps']
    steps_text = ""
    for i, step in enumerate(steps):
        steps_text += f"Step {i + 1}: {step}\n"

    question = sample['binary_question']

    system_msg = (
        "You are an expert at understanding procedural steps and their dependencies. "
        "Given a set of recipe steps and a question about their ordering, reason about "
        "whether the ordering constraint must hold, then answer with Yes or No."
    )

    user_msg = (
        f"Here are the steps of a recipe:\n\n{steps_text}\n"
        f"{question}\n\n"
        f"Think step by step, then answer Yes or No."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    return prompt_text


def build_eval_prompt_with_why(sample, tokenizer):
    """
    Build a chat prompt that asks both the binary question and the why question.
    """
    steps = sample['steps']
    steps_text = ""
    for i, step in enumerate(steps):
        steps_text += f"Step {i + 1}: {step}\n"

    binary_q = sample['binary_question']
    why_q = sample['why_question']

    system_msg = (
        "You are an expert at understanding procedural steps and their dependencies. "
        "Given a set of recipe steps and questions about their ordering, reason about "
        "the dependencies between steps."
    )

    user_msg = (
        f"Here are the steps of a recipe:\n\n{steps_text}\n"
        f"Question 1: {binary_q}\n"
        f"Question 2: {why_q}\n\n"
        f"Think step by step, then provide:\n"
        f"1. Answer to Question 1 (Yes or No)\n"
        f"2. Answer to Question 2 (a brief explanation)"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    return prompt_text


# ======================================================================
# Answer parsing
# ======================================================================

def parse_binary_answer(response_text):
    """
    Parse yes/no from model response.
    Looks for the answer after </think> if present.

    Returns:
        1 for yes, 0 for no, None if unparseable
    """
    # Look after </think> if present
    think_end = response_text.find("</think>")
    if think_end != -1:
        answer_part = response_text[think_end + len("</think>"):]
    else:
        answer_part = response_text

    answer_part = answer_part.strip().lower()

    # Check for explicit yes/no patterns
    # Priority: look for "answer: yes/no" or "1. yes/no" patterns first
    explicit = re.search(r'(?:answer[:\s]*|1[.)\s]+)\s*(yes|no)\b', answer_part)
    if explicit:
        return 1 if explicit.group(1) == 'yes' else 0

    # Fallback: first yes/no in the answer portion
    match = re.search(r'\b(yes|no)\b', answer_part)
    if match:
        return 1 if match.group(1) == 'yes' else 0

    # Last resort: check the very last non-empty line
    lines = [l.strip() for l in answer_part.split('\n') if l.strip()]
    if lines:
        last = lines[-1].lower()
        if 'yes' in last:
            return 1
        elif 'no' in last:
            return 0

    return None


# ======================================================================
# Main evaluation
# ======================================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    ).to(device)
    model.eval()

    # Load test data
    print(f"Loading data: {args.data_path}", flush=True)
    with open(args.data_path, "r", encoding="utf8") as f:
        data = json.load(f)

    if args.num_samples > 0:
        data = data[:args.num_samples]

    print(f"Evaluating {len(data)} samples...", flush=True)

    results = []
    correct = 0
    total = 0
    parse_failures = 0

    # Per-type tracking
    type_correct = {}
    type_total = {}

    for sample in tqdm(data, desc="Evaluating"):
        if args.include_why:
            prompt_text = build_eval_prompt_with_why(sample, tokenizer)
        else:
            prompt_text = build_eval_prompt(sample, tokenizer)

        inputs = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(device)
        prompt_len = inputs['input_ids'].shape[1]

        if prompt_len > args.max_prompt_length:
            continue

        with torch.no_grad():
            # Qwen3 thinking mode: do NOT use greedy decoding (causes
            # performance degradation and endless repetitions).
            # Recommended: temperature=0.6, top_p=0.95, top_k=20
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = output_ids[0, prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

        predicted = parse_binary_answer(response_text)
        label = sample['label']
        question_type = sample.get('question_type', 'unknown')

        result = {
            'plan_idx': sample.get('plan_idx'),
            'question_idx': sample.get('question_idx'),
            'question_type': question_type,
            'binary_question': sample['binary_question'],
            'label': label,
            'predicted': predicted,
            'correct': predicted == label if predicted is not None else False,
            'response': response_text[:1000],
        }
        results.append(result)

        if predicted is not None:
            total += 1
            if predicted == label:
                correct += 1

            if question_type not in type_correct:
                type_correct[question_type] = 0
                type_total[question_type] = 0
            type_total[question_type] += 1
            if predicted == label:
                type_correct[question_type] += 1
        else:
            parse_failures += 1

    # ---- Print results ----------------------------------------------------
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Parse failures: {parse_failures}")
    print(f"\nPer-type breakdown:")
    for qtype in sorted(type_total.keys()):
        acc = type_correct[qtype] / type_total[qtype] if type_total[qtype] > 0 else 0.0
        print(f"  {qtype}: {acc:.4f} ({type_correct[qtype]}/{type_total[qtype]})")
    print(f"{'=' * 60}")

    # ---- Save results -----------------------------------------------------
    if args.output_path:
        output = {
            'model_name': args.model_name,
            'data_path': args.data_path,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'parse_failures': parse_failures,
            'per_type': {
                qtype: {
                    'accuracy': type_correct[qtype] / type_total[qtype]
                               if type_total[qtype] > 0 else 0.0,
                    'correct': type_correct[qtype],
                    'total': type_total[qtype],
                }
                for qtype in sorted(type_total.keys())
            },
            'results': results,
        }
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate on CaT-Bench test_must_why in natural language"
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--data_path",
        default="./data/cat_bench/catplan-data-release/generated_questions/"
                "test_must_why/test_must_why.json",
    )
    parser.add_argument("--output_path", default="./results/cat_bench_eval.json")
    parser.add_argument("--num_samples", default=0, type=int,
                        help="0 = evaluate all samples")
    parser.add_argument("--max_new_tokens", default=32768, type=int,
                        help="Qwen3 recommends 32768 for thinking mode")
    parser.add_argument("--max_prompt_length", default=2048, type=int)
    parser.add_argument("--bf16", default=1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float,
                        help="Qwen3 thinking mode recommended: 0.6")
    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Qwen3 thinking mode recommended: 0.95")
    parser.add_argument("--top_k", default=20, type=int,
                        help="Qwen3 thinking mode recommended: 20")
    parser.add_argument("--include_why", default=0, type=int,
                        help="Also ask the why question (richer prompt)")

    args = parser.parse_args()
    main(args)
