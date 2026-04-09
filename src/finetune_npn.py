"""
Fine-tune and evaluate GPT-2 on the NPN cooking dataset tasks from:
  Bosselut et al. (2018) "Simulating Action Dynamics with Neural Process Networks"

Each recipe is formatted as an interleaved sequence:

    ingredients: lobster, sugar, fish_sauce, ...
    step 1: put the oil in a wok over medium heat .
    entities: vegetable_oil
    changes: location: wok; temperature: hot
    step 2: fry the garlic , shallots , and coriander root .
    entities: garlic, coriander, shallot
    changes: cookedness: cooked; temperature: hot
    ...

Training: loss is masked on everything except "entities:" and "changes:" lines.
Inference: generate step-by-step, feeding each prediction back as context.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_model import load_model_from_checkpoint
from tqdm.auto import tqdm
import argparse
import json
import os
import random
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

STATE_CHANGE_TYPES = [
    "location", "cookedness", "temperature", "composition",
    "shape", "cleanliness", "rotation", "accessibility",
]

ORIGINAL_PAPER_TYPES = {
    "location", "cookedness", "temperature", "composition",
    "shape", "cleanliness",
}
ALL_TYPES = set(STATE_CHANGE_TYPES)

ENTITIES_PREFIX = "entities:"
CHANGES_PREFIX = "changes:"


def load_npn_data(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    splits = defaultdict(list)
    for d in data:
        splits[d["split"]].append(d)
    return splits


def _join_tokens(token_list):
    return " ".join(token_list)


# ──────────────────────────────────────────────────────────────────────────────
# Build full recipe sequences (one per recipe)
# ──────────────────────────────────────────────────────────────────────────────

def _build_step_annotations(doc, step_idx):
    """Build the gold entities and changes strings for a single step."""
    step_key = str(step_idx)
    ingredient_list = doc["ingredient_list"]
    ingredients = doc["ingredients"]
    events = doc.get("events", {})

    # Entities
    gold_ent_indices = ingredients.get(step_key, [])
    gold_ent_names = sorted(set(
        ingredient_list[idx] for idx in gold_ent_indices
        if idx < len(ingredient_list)
    ))
    entities_str = ", ".join(gold_ent_names) if gold_ent_names else "none"

    # State changes
    step_events = events.get(step_key, {})
    change_parts = []
    for sc_type in STATE_CHANGE_TYPES:
        if sc_type in step_events:
            for es in step_events[sc_type]:
                change_parts.append(f"{sc_type}: {es}")
    changes_str = "; ".join(change_parts) if change_parts else "none"

    return entities_str, changes_str, gold_ent_names, step_events


def build_recipe_sequence(doc):
    """
    Build the full interleaved text for one recipe, plus per-step metadata.

    Returns:
        full_text: the entire sequence string
        step_info: list of dicts with gold labels and char offsets for
                   the answer portions (entities + changes lines)
    """
    ingredient_list = doc["ingredient_list"]
    sentences = doc["text"]
    n_steps = len(sentences)

    ing_str = ", ".join(ingredient_list)
    parts = [f"ingredients: {ing_str}"]
    step_info = []

    for step_idx in range(n_steps):
        sent = _join_tokens(sentences[str(step_idx)])
        entities_str, changes_str, gold_ents, gold_events = _build_step_annotations(doc, step_idx)

        parts.append(f"step {step_idx + 1}: {sent}")
        parts.append(f"{ENTITIES_PREFIX} {entities_str}")
        parts.append(f"{CHANGES_PREFIX} {changes_str}")

        step_info.append({
            "step_idx": step_idx,
            "gold_entities": gold_ents,
            "gold_events": gold_events,
            "entities_str": entities_str,
            "changes_str": changes_str,
        })

    full_text = "\n".join(parts)
    return full_text, step_info


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: one sample = one full recipe, loss masked to annotation lines only
# ──────────────────────────────────────────────────────────────────────────────

def _compute_answer_mask(full_text, tokenizer, max_length):
    """
    Tokenize the full recipe and build a label mask that only includes
    the tokens belonging to 'entities: ...' and 'changes: ...' lines.
    """
    enc = tokenizer(
        full_text, truncation=True, max_length=max_length, return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)

    # Decode token by token to find answer regions
    # Strategy: find the character spans of answer lines, then map to token indices
    labels = torch.full_like(input_ids, -100)

    # Find all answer line spans in the original text
    lines = full_text.split("\n")
    char_pos = 0
    answer_char_ranges = []
    for line in lines:
        line_start = char_pos
        line_end = char_pos + len(line)
        if line.startswith(ENTITIES_PREFIX) or line.startswith(CHANGES_PREFIX):
            # The answer portion is after the prefix (include the space + content)
            prefix = ENTITIES_PREFIX if line.startswith(ENTITIES_PREFIX) else CHANGES_PREFIX
            answer_start = line_start + len(prefix)
            answer_char_ranges.append((answer_start, line_end))
        char_pos = line_end + 1  # +1 for the \n

    # Map character ranges to token indices using offset_mapping
    enc_with_offsets = tokenizer(
        full_text, truncation=True, max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc_with_offsets["offset_mapping"]

    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for ans_start, ans_end in answer_char_ranges:
            if tok_start >= ans_start and tok_end <= ans_end:
                labels[tok_idx] = input_ids[tok_idx]
                break

    return input_ids, attention_mask, labels


class NPNCookingDataset(Dataset):
    """Lazy-tokenizing dataset — one sample per recipe."""

    def __init__(self, recipes, tokenizer, max_length):
        self.recipes = recipes
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-build text sequences (cheap, just string ops)
        self.sequences = []
        for doc in recipes:
            full_text, step_info = build_recipe_sequence(doc)
            self.sequences.append(full_text)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_text = self.sequences[idx]
        input_ids, attention_mask, labels = _compute_answer_mask(
            full_text, self.tokenizer, self.max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        attention_mask[i, :n] = x["attention_mask"]
        labels[i, :n] = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# Inference: step-by-step generation with feedback
# ──────────────────────────────────────────────────────────────────────────────

def _generate_until_newline(model, tokenizer, prompt, device, max_length, max_gen_tokens,
                            newline_id, eos_id):
    """Generate tokens until newline or eos, return decoded text (stripped)."""
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    )["input_ids"].to(device)

    stop_ids = list(set([newline_id, eos_id]))
    output = model.generate(
        input_ids,
        max_new_tokens=max_gen_tokens,
        do_sample=False,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_tokens = output[0][input_ids.shape[1]:]
    # Truncate at first newline or eos if present in the generated tokens
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    # Take only the first line
    text = text.split("\n")[0].strip()
    return text


def generate_recipe_predictions(model, tokenizer, doc, device, max_length=1024, max_gen_tokens=64):
    """
    Process one recipe step-by-step:
      1. Feed context up to "step N: <sentence>\nentities:"
      2. Generate entities prediction, stop at newline
      3. Append prediction, feed up to "changes:"
      4. Generate changes prediction, stop at newline
      5. Append and continue to next step

    Returns list of (pred_entities_str, pred_changes_str) per step.
    """
    model.eval()
    ingredient_list = doc["ingredient_list"]
    sentences = doc["text"]
    n_steps = len(sentences)
    ing_str = ", ".join(ingredient_list)

    context = f"ingredients: {ing_str}"
    predictions = []

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for step_idx in range(n_steps):
            sent = _join_tokens(sentences[str(step_idx)])
            context += f"\nstep {step_idx + 1}: {sent}"

            # Generate entities
            prompt_ent = context + f"\n{ENTITIES_PREFIX}"
            pred_entities = _generate_until_newline(
                model, tokenizer, prompt_ent, device, max_length, max_gen_tokens,
                newline_id, eos_id,
            )

            # Generate changes
            context += f"\n{ENTITIES_PREFIX} {pred_entities}"
            prompt_chg = context + f"\n{CHANGES_PREFIX}"
            pred_changes = _generate_until_newline(
                model, tokenizer, prompt_chg, device, max_length, max_gen_tokens,
                newline_id, eos_id,
            )

            context += f"\n{CHANGES_PREFIX} {pred_changes}"
            predictions.append((pred_entities, pred_changes))

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

def _parse_entity_set(text):
    """Parse comma-separated entity list."""
    text = text.strip().lower()
    if not text or text == "none":
        return set()
    return set(e.strip() for e in text.split(",") if e.strip())


def _parse_state_changes(text):
    """Parse 'type: end_state; type: end_state' string."""
    text = text.strip().lower()
    if not text or text == "none":
        return {}
    result = defaultdict(list)
    for part in text.split(";"):
        part = part.strip()
        if ":" in part:
            sc_type, end_state = part.split(":", 1)
            sc_type = sc_type.strip()
            end_state = end_state.strip()
            if sc_type and end_state:
                result[sc_type].append(end_state)
    return dict(result)


def _compute_sc_metrics(gold_types, pred_types, gold_events, pred_events, type_filter=None):
    """Compute state change type F1 and end state accuracy for a single step.
    If type_filter is given, only consider types in that set."""
    if type_filter is not None:
        gold_types = gold_types & type_filter
        pred_types = pred_types & type_filter

    tp = len(gold_types & pred_types)
    fp = len(pred_types - gold_types)
    fn = len(gold_types - pred_types)

    es_correct = es_total = 0
    for sc_type in gold_types & pred_types:
        gold_es = set(e.lower() for e in gold_events[sc_type])
        pred_es = set(e.lower() for e in pred_events.get(sc_type, []))
        es_total += len(gold_es)
        es_correct += len(gold_es & pred_es)

    return tp, fp, fn, es_correct, es_total


def _f1_from_counts(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_predictions(eval_data, all_predictions):
    """
    Evaluate entity selection and state change predictions across all recipes.

    Reports two sets of state-change metrics:
      - all_*:           over all 8 state change types in the data
      - original_paper_*: over the 6 types from the original paper only

    Args:
        eval_data: list of recipe dicts
        all_predictions: list of lists of (pred_entities_str, pred_changes_str)

    Returns dict with metrics.
    """
    # Entity selection
    ent_tp = ent_fp = ent_fn = 0
    # State changes — all types
    all_sc_tp = all_sc_fp = all_sc_fn = 0
    all_es_correct = all_es_total = 0
    # State changes — original paper types only
    op_sc_tp = op_sc_fp = op_sc_fn = 0
    op_es_correct = op_es_total = 0

    for doc, preds in zip(eval_data, all_predictions):
        n_steps = len(doc["text"])
        for step_idx in range(min(n_steps, len(preds))):
            _, _, gold_ents, gold_events = _build_step_annotations(doc, step_idx)
            pred_entities_str, pred_changes_str = preds[step_idx]

            # Entity selection
            gold_ent_set = set(e.lower() for e in gold_ents)
            pred_ent_set = _parse_entity_set(pred_entities_str)
            ent_tp += len(gold_ent_set & pred_ent_set)
            ent_fp += len(pred_ent_set - gold_ent_set)
            ent_fn += len(gold_ent_set - pred_ent_set)

            # State changes
            gold_types = set(gold_events.keys())
            pred_events = _parse_state_changes(pred_changes_str)
            pred_types = set(pred_events.keys())

            # All types
            tp, fp, fn, esc, est = _compute_sc_metrics(
                gold_types, pred_types, gold_events, pred_events, type_filter=None,
            )
            all_sc_tp += tp; all_sc_fp += fp; all_sc_fn += fn
            all_es_correct += esc; all_es_total += est

            # Original paper types only
            tp, fp, fn, esc, est = _compute_sc_metrics(
                gold_types, pred_types, gold_events, pred_events,
                type_filter=ORIGINAL_PAPER_TYPES,
            )
            op_sc_tp += tp; op_sc_fp += fp; op_sc_fn += fn
            op_es_correct += esc; op_es_total += est

    ent_prec, ent_rec, ent_f1 = _f1_from_counts(ent_tp, ent_fp, ent_fn)
    all_sc_prec, all_sc_rec, all_sc_f1 = _f1_from_counts(all_sc_tp, all_sc_fp, all_sc_fn)
    op_sc_prec, op_sc_rec, op_sc_f1 = _f1_from_counts(op_sc_tp, op_sc_fp, op_sc_fn)
    all_es_acc = all_es_correct / all_es_total if all_es_total > 0 else 0.0
    op_es_acc = op_es_correct / op_es_total if op_es_total > 0 else 0.0

    return {
        "entity_f1": ent_f1,
        "entity_precision": ent_prec,
        "entity_recall": ent_rec,
        # All 8 state change types
        "all_sc_type_f1": all_sc_f1,
        "all_sc_type_precision": all_sc_prec,
        "all_sc_type_recall": all_sc_rec,
        "all_endstate_accuracy": all_es_acc,
        "all_endstate_total": all_es_total,
        # Original paper 6 types only
        "original_paper_sc_type_f1": op_sc_f1,
        "original_paper_sc_type_precision": op_sc_prec,
        "original_paper_sc_type_recall": op_sc_rec,
        "original_paper_endstate_accuracy": op_es_acc,
        "original_paper_endstate_total": op_es_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, epoch, max_steps=0, global_step=0):
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if max_steps > 0 and global_step >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def run_evaluation(model, tokenizer, eval_data, device, max_length=1024):
    """Run step-by-step generation on all recipes and compute metrics."""
    all_predictions = []
    for doc in tqdm(eval_data, desc="Evaluating"):
        preds = generate_recipe_predictions(
            model, tokenizer, doc, device,
            max_length=max_length,
        )
        all_predictions.append(preds)

    metrics = evaluate_predictions(eval_data, all_predictions)
    return metrics, all_predictions


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic seeding
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    splits = load_npn_data(args.data_path)
    train_data = splits["train"]
    dev_data = splits["dev"]
    test_data = splits["test"]

    if args.max_recipes > 0:
        train_data = train_data[:args.max_recipes]
    if args.max_eval_recipes > 0:
        dev_data = dev_data[:args.max_eval_recipes]
        test_data = test_data[:args.max_eval_recipes]

    print(f"Data — train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}", flush=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model_from_checkpoint(args.model_dir, device=device)
    print(f"Model: {args.model_dir}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Derive output dirs
    model_name_path = os.path.normpath(args.model_dir)
    sub_path = ""
    parts = model_name_path.split(os.sep)
    if "models" in parts:
        idx = parts.index("models")
        if len(parts) > idx + 2:
            sub_path = os.path.join(*parts[idx+2:])
    else:
        sub_path = model_name_path.replace(os.sep, "_")

    seed_leaf = f"seed={args.seed}"
    if sub_path:
        output_dir = os.path.join(args.output_dir, sub_path, seed_leaf)
        results_dir = os.path.join(args.results_dir, sub_path, seed_leaf)
    else:
        output_dir = os.path.join(args.output_dir, seed_leaf)
        results_dir = os.path.join(args.results_dir, seed_leaf)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    max_length = getattr(model.config, "max_position_embeddings", 1024)

    # Show a sample recipe sequence
    sample_text, sample_info = build_recipe_sequence(train_data[0])
    print("\n=== Sample recipe sequence ===", flush=True)
    # Show first ~20 lines
    for line in sample_text.split("\n")[:20]:
        print(line, flush=True)
    if len(sample_text.split("\n")) > 20:
        print("...", flush=True)
    print("=" * 60, flush=True)

    if not args.eval_only:
        # Training
        dataset = NPNCookingDataset(train_data, tokenizer, max_length=max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )

        # Show first batch label masking
        first_batch = next(iter(dataloader))
        print("\n=== First batch: input vs labels ===", flush=True)
        n_show = min(2, first_batch["input_ids"].shape[0])
        for i in range(n_show):
            decoded_input = tokenizer.decode(first_batch["input_ids"][i], skip_special_tokens=False)
            label_ids = first_batch["labels"][i]
            decoded_labels = tokenizer.decode(label_ids[label_ids != -100], skip_special_tokens=False)
            print(f"--- Recipe {i} ---", flush=True)
            print(f"Full text (first 300 chars): {decoded_input[:300]}", flush=True)
            print(f"Labels (supervised tokens): {decoded_labels[:300]}", flush=True)
        print("=" * 60, flush=True)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            avg_loss, global_step = train_epoch(
                model, dataloader, optimizer, device,
                epoch=epoch, max_steps=args.max_steps, global_step=global_step,
            )
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}", flush=True)

            # Dev evaluation
            dev_metrics, _ = run_evaluation(
                model, tokenizer, dev_data, device, max_length=max_length,
            )
            print(
                f"Dev (epoch {epoch}) — "
                f"ent_F1: {dev_metrics['entity_f1']:.4f} | "
                f"all_sc_F1: {dev_metrics['all_sc_type_f1']:.4f} | "
                f"all_es_acc: {dev_metrics['all_endstate_accuracy']:.4f} | "
                f"paper_sc_F1: {dev_metrics['original_paper_sc_type_f1']:.4f} | "
                f"paper_es_acc: {dev_metrics['original_paper_endstate_accuracy']:.4f}",
                flush=True,
            )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}, stopping.", flush=True)
                break

        # Save model
        final_dir = os.path.join(output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Model saved: {final_dir}", flush=True)

    # Test evaluation
    print("\n=== Test Evaluation ===", flush=True)
    test_metrics, test_preds = run_evaluation(
        model, tokenizer, test_data, device, max_length=max_length,
    )
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}", flush=True)

    # Save results
    results_path = os.path.join(results_dir, "test_results.json")
    save_data = {
        "model": args.model_dir,
        "metrics": test_metrics,
        "predictions": [],
    }
    for doc, preds in zip(test_data, test_preds):
        for step_idx, (pred_ent, pred_chg) in enumerate(preds):
            _, _, gold_ents, gold_events = _build_step_annotations(doc, step_idx)
            save_data["predictions"].append({
                "doc_id": doc["id"],
                "step_idx": step_idx,
                "pred_entities": pred_ent,
                "gold_entities": gold_ents,
                "pred_changes": pred_chg,
                "gold_events": gold_events,
            })
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved: {results_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on NPN Cooking (entity selection + state changes)"
    )
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--data_path", default="./data/npn_cooking/npn_data.json")
    parser.add_argument("--output_dir", default="./models/npn_cooking")
    parser.add_argument("--results_dir", default="./results/npn_cooking")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=0, type=int, help="0 = no limit")
    parser.add_argument("--max_recipes", default=0, type=int, help="Limit train recipes (0 = all)")
    parser.add_argument("--max_eval_recipes", default=0, type=int, help="Limit eval recipes (0 = all)")
    parser.add_argument("--eval_only", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)
