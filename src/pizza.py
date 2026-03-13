import argparse
import json
import random
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextStreamer
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_fn
# Monkey-patch: bert_score calls build_inputs_with_special_tokens which was
# removed in newer transformers. Patch sent_encode to use a simple .encode().
import bert_score.utils as _bsu
_orig_sent_encode = _bsu.sent_encode
def _patched_sent_encode(tokenizer, text):
    try:
        return _orig_sent_encode(tokenizer, text)
    except AttributeError:
        return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length)
_bsu.sent_encode = _patched_sent_encode
from tqdm.auto import tqdm
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Special tokens following the paper's serialization format (Figure 2)
SEP_TOKEN = " <s> "    # separates cells within a row
ROW_TOKEN = " <r>\n"   # separates rows
MASK_IN = "<in>"        # mask token for input
MASK_OUT = "<out>"      # mask token for output


def get_action(step):
    """Handle inconsistent key: some recipes use 'action', others 'actions'."""
    return step.get("action", step.get("actions", "NA"))


def serialize_recipe(recipe, masked=False):
    """Serialize a recipe table into text.

    Full format per row: instructions <s> input <s> action <s> output <r>
    Masked format: instructions <s> <in> <s> action <s> <out> <r>
    """
    title = recipe.get("title", "")
    lines = [f"Title: {title}\n"]
    for step in recipe.get("table", []):
        instruction = step.get("instructions", "NA")
        inp = step.get("input", "NA") if not masked else MASK_IN
        action = get_action(step)
        output = step.get("output", "NA") if not masked else MASK_OUT
        lines.append(f"{instruction}{SEP_TOKEN}{inp}{SEP_TOKEN}{action}{SEP_TOKEN}{output}{ROW_TOKEN}")
    return "".join(lines)


def format_training_data(data_path, tokenizer=None, use_chat_template=False, enable_thinking=False):
    """Format data for CLM training: full serialized tables."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    for recipe in data:
        text = serialize_recipe(recipe, masked=False)
        if use_chat_template and tokenizer is not None:
            messages = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": text},
            ]
            kwargs = {"tokenize": False, "add_generation_prompt": False}
            if enable_thinking:
                kwargs["enable_thinking"] = True
            else:
                kwargs["enable_thinking"] = False
            text = tokenizer.apply_chat_template(messages, **kwargs)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def compute_metrics(gold_inputs, pred_inputs, gold_outputs, pred_outputs):
    """Compute Table 1 metrics.

    Input metrics: EMA, ROUGE_L, BLEU
    Output metrics: ROUGE_L, METEOR, BERTScore
    """
    n = len(gold_inputs)
    assert n == len(pred_inputs) == len(gold_outputs) == len(pred_outputs)

    # --- Input metrics ---
    # EMA (Exact Matching Accuracy)
    input_ema = sum(1 for g, p in zip(gold_inputs, pred_inputs) if g.strip().lower() == p.strip().lower()) / n * 100

    # ROUGE_L for inputs
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    input_rouge_scores = [scorer.score(g, p)['rougeL'].fmeasure for g, p in zip(gold_inputs, pred_inputs)]
    input_rouge = np.mean(input_rouge_scores) * 100

    # BLEU for inputs
    smooth = SmoothingFunction().method1
    input_bleu_scores = []
    for g, p in zip(gold_inputs, pred_inputs):
        ref_tokens = g.lower().split()
        hyp_tokens = p.lower().split()
        if len(hyp_tokens) == 0:
            input_bleu_scores.append(0.0)
        else:
            input_bleu_scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth))
    input_bleu = np.mean(input_bleu_scores) * 100

    # --- Output metrics ---
    # ROUGE_L for outputs
    output_rouge_scores = [scorer.score(g, p)['rougeL'].fmeasure for g, p in zip(gold_outputs, pred_outputs)]
    output_rouge = np.mean(output_rouge_scores) * 100

    # METEOR for outputs
    output_meteor_scores = []
    for g, p in zip(gold_outputs, pred_outputs):
        ref_tokens = g.lower().split()
        hyp_tokens = p.lower().split()
        output_meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
    output_meteor = np.mean(output_meteor_scores) * 100

    # BERTScore for outputs
    P, R, F1 = bert_score_fn(pred_outputs, gold_outputs, lang="en", verbose=False)
    output_bertscore = F1.mean().item() * 100

    return {
        "Input_EMA": input_ema,
        "Input_ROUGE_L": input_rouge,
        "Input_BLEU": input_bleu,
        "Output_ROUGE_L": output_rouge,
        "Output_METEOR": output_meteor,
        "Output_BERTScore": output_bertscore,
    }


CHAT_SYSTEM_PROMPT = (
    "You are tracking ingredient states in a recipe. "
    "Each step is formatted as: instruction <s> input <s> action <s> output <r>\n"
    "Given the recipe history and a new instruction, predict the remaining cells: input <s> action <s> output"
)


def _build_icl_prefix(train_data, n_icl):
    """Build an ICL prefix from n_icl randomly sampled training recipes."""
    if n_icl <= 0 or not train_data:
        return ""
    samples = random.sample(train_data, min(n_icl, len(train_data)))
    parts = []
    for i, recipe in enumerate(samples, 1):
        parts.append(f"Example {i}:\n{serialize_recipe(recipe, masked=False)}\n")
    parts.append("Given the instructions and examples above, carry out the task on the text below:\n\n")
    return "".join(parts)


def build_prompts(val_data, tokenizer=None, use_chat_template=False, enable_thinking=False, train_data=None, n_icl=0):
    """Pre-build all prompts and collect gold labels.

    If use_chat_template=True, wraps prompts with apply_chat_template for
    instruction-tuned / chat models (e.g. Llama-3-Instruct, Qwen2.5-Instruct).
    If n_icl > 0, prepends n_icl randomly sampled training recipes as ICL examples.
    """
    icl_prefix = _build_icl_prefix(train_data, n_icl)

    prompts = []
    gold_inputs = []
    gold_outputs = []

    for recipe in val_data:
        steps = recipe.get("table", [])
        title = recipe.get("title", "")

        for step_idx, step in enumerate(steps):
            prompt_parts = [f"Title: {title}\n"]
            for prev_idx in range(step_idx):
                prev = steps[prev_idx]
                prompt_parts.append(
                    f"{prev['instructions']}{SEP_TOKEN}{prev['input']}{SEP_TOKEN}"
                    f"{get_action(prev)}{SEP_TOKEN}{prev['output']}{ROW_TOKEN}"
                )
            prompt_parts.append(f"{step['instructions']}{SEP_TOKEN}")
            raw_prompt = icl_prefix + "".join(prompt_parts)

            if use_chat_template and tokenizer is not None:
                messages = [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_prompt},
                ]
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if enable_thinking:
                    kwargs["enable_thinking"] = True
                else:
                    kwargs["enable_thinking"] = False
                raw_prompt = tokenizer.apply_chat_template(messages, **kwargs)

            prompts.append(raw_prompt)
            gold_inputs.append(step.get("input", "NA"))
            gold_outputs.append(step.get("output", "NA"))

    return prompts, gold_inputs, gold_outputs


def evaluate_generation(model, tokenizer, val_data, device, max_new_tokens=128, batch_size=16, num_samples=0, use_chat_template=False, enable_thinking=False, verbose_streamer=False, train_data=None, n_icl=0):
    """Evaluate with batched generation for speed.

    For each step, the prompt contains prior steps (full) + current instruction.
    The model generates: input <s> action <s> output <r>
    """
    model.eval()

    prompts, all_gold_inputs, all_gold_outputs = build_prompts(
        val_data, tokenizer=tokenizer, use_chat_template=use_chat_template,
        enable_thinking=enable_thinking, train_data=train_data, n_icl=n_icl
    )
    if num_samples > 0:
        prompts = prompts[:num_samples]
        all_gold_inputs = all_gold_inputs[:num_samples]
        all_gold_outputs = all_gold_outputs[:num_samples]
    all_pred_inputs = []
    all_pred_outputs = []

    # Stop generation at <r> to avoid running past the current step
    r_token_id = tokenizer.convert_tokens_to_ids("<r>")
    eos_ids = [tokenizer.eos_token_id]
    if r_token_id != tokenizer.unk_token_id:
        eos_ids.append(r_token_id)

    tokenizer.padding_side = "left"
    streamer = TextStreamer(tokenizer, skip_special_tokens=False) if verbose_streamer else None

    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        encoded = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                streamer=streamer,
            )

        # Slice off the prompt tokens to get only the generated continuation
        prompt_len = encoded["input_ids"].shape[1]
        gen_ids = output_ids[:, prompt_len:]

        for i in range(len(batch_prompts)):
            # Don't skip special tokens — we need <s> and <r> for parsing
            continuation = tokenizer.decode(gen_ids[i], skip_special_tokens=False)
            # Remove EOS/PAD tokens but keep our delimiters
            for tok in [tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token]:
                if tok:
                    continuation = continuation.replace(tok, "")
            # Strip thinking tokens from reasoning models (e.g. QwQ, DeepSeek-R1)
            if "</think>" in continuation:
                continuation = continuation.split("</think>")[-1]

            # Debug: print first 5 raw continuations
            # if len(all_pred_inputs) < 5:
            #     print(f"  [DEBUG] raw continuation: {repr(continuation)}")

            # Parse: we expect "input <s> action <s> output <r>"
            parts = continuation.split("<s>")
            pred_input = parts[0].strip() if len(parts) > 0 else ""
            pred_output = parts[2].strip().split("<r>")[0].strip() if len(parts) > 2 else ""

            all_pred_inputs.append(pred_input)
            all_pred_outputs.append(pred_output)

    metrics = compute_metrics(all_gold_inputs, all_pred_inputs, all_gold_outputs, all_pred_outputs)
    return metrics, all_gold_inputs, all_pred_inputs, all_gold_outputs, all_pred_outputs


def main(args):
    # Load validation data (raw JSON for evaluation)
    with open(args.val_file, 'r', encoding='utf-8') as f:
        val_data_raw = json.load(f)

    # Load training data for ICL examples
    train_data_raw = None
    if args.n_icl > 0:
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_data_raw = json.load(f)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    if not args.eval_only:
        # --- Training ---
        train_dataset = format_training_data(args.train_file, tokenizer=tokenizer, use_chat_template=args.use_chat_template, enable_thinking=args.enable_thinking)
        val_dataset = format_training_data(args.val_file, tokenizer=tokenizer, use_chat_template=args.use_chat_template, enable_thinking=args.enable_thinking)
        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            push_to_hub=False,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        print("Saving model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # --- Evaluation (Table 1 metrics) ---
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("\n" + "=" * 60)
    print("Evaluating with Table 1 metrics (PizzaCommonSense)")
    print("=" * 60)

    metrics, gold_in, pred_in, gold_out, pred_out = evaluate_generation(
        model, tokenizer, val_data_raw, device,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        num_samples=args.num_samples, use_chat_template=args.use_chat_template,
        enable_thinking=args.enable_thinking, verbose_streamer=args.verbose_streamer,
        train_data=train_data_raw, n_icl=args.n_icl
    )

    # Print results in Table 1 format
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Value':>8}")
    print("-" * 30)
    print("--- Input ---")
    print(f"  {'EMA':<18} {metrics['Input_EMA']:>7.1f}")
    print(f"  {'ROUGE_L':<18} {metrics['Input_ROUGE_L']:>7.1f}")
    print(f"  {'BLEU':<18} {metrics['Input_BLEU']:>7.1f}")
    print("--- Output ---")
    print(f"  {'ROUGE_L':<18} {metrics['Output_ROUGE_L']:>7.1f}")
    print(f"  {'METEOR':<18} {metrics['Output_METEOR']:>7.1f}")
    print(f"  {'BERTScore':<18} {metrics['Output_BERTScore']:>7.1f}")
    print("=" * 60)

    # Print example predictions
    print(f"\nExample predictions (first {args.num_examples}):")
    for i in range(min(args.num_examples, len(gold_in))):
        print(f"\n--- Step {i+1} ---")
        print(f"  Gold Input:    {gold_in[i]}")
        print(f"  Pred Input:    {pred_in[i]}")
        print(f"  Gold Output:   {gold_out[i]}")
        print(f"  Pred Output:   {pred_out[i]}")

    # Save metrics to JSON
    metrics_path = f"{args.output_dir}/eval_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Evaluate on PizzaCommonSense (Table 1 metrics)")
    parser.add_argument("--train_file", type=str, default="data/pizza_common_sense/train.json")
    parser.add_argument("--val_file", type=str, default="data/pizza_common_sense/val.json")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2")
    parser.add_argument("--output_dir", type=str, default="results/pizza_common_sense")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_only", type=int, default=1, help="Skip training, only evaluate (1=True, 0=False)")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to print")
    parser.add_argument("--num_samples", type=int, default=100, help="Max eval steps (0 = all)")
    parser.add_argument("--use_chat_template", type=int, default=1, help="Wrap prompts with apply_chat_template for chat/instruct models (1=True, 0=False)")
    parser.add_argument("--enable_thinking", type=int, default=1, help="Pass enable_thinking to apply_chat_template (1=True, 0=False; for reasoning models like QwQ)")
    parser.add_argument("--verbose_streamer", type=int, default=1, help="Stream generated tokens to stdout during evaluation (1=True, 0=False)")
    parser.add_argument("--n_icl", type=int, default=0, help="Number of training recipes to include as ICL examples in the prompt (0=none)")
    args = parser.parse_args()
    main(args)
