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

# NLTK utilities for Cat-3 location matching (stop-word removal + lemmatizing)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
    try:
        _LEMMATIZER = WordNetLemmatizer()
        _LEMMATIZER.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    STOP_WORDS = set()
    _LEMMATIZER = None

STATE_MAP = {"-": "-", "?": "?"}
NONE_VALS = {"none", "", "-"}
UNKNOWN_VALS = {"unknown", "?"}


def load_propara_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def map_state(state):
    state = str(state).strip().lower()
    return STATE_MAP.get(state, state)


def compute_state_change(prev_state, curr_state):
    """Derive state change type from two consecutive mapped states."""
    prev = prev_state.strip().lower()
    curr = curr_state.strip().lower()
    if prev in UNKNOWN_VALS or curr in UNKNOWN_VALS:
        return "none"
    prev_absent = prev in NONE_VALS
    curr_absent = curr in NONE_VALS
    if prev_absent and not curr_absent:
        return "create"
    elif not prev_absent and curr_absent:
        return "destroy"
    elif not prev_absent and not curr_absent and prev != curr:
        return "move"
    else:
        return "none"


def format_answer(curr_state):
    """Encode next-state target string for training/decoding."""
    return curr_state


def parse_prediction(pred_str):
    """Parse a model-generated next-state string (first non-empty line)."""
    text = pred_str.strip().lower()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if first_line.startswith("state:"):
        first_line = first_line[len("state:") :].strip()
    return map_state(first_line)


def normalize_loc(loc):
    """Lowercase, remove stop words, lemmatize — used for Cat-3 matching."""
    tokens = loc.lower().split()
    if _NLTK_AVAILABLE and _LEMMATIZER is not None:
        tokens = [
            _LEMMATIZER.lemmatize(t)
            for t in tokens
            if t not in STOP_WORDS and t.isalpha()
        ]
    return set(tokens)


def location_match(pred_loc, gold_loc):
    """
    True if pred_loc is identical to, or a sub-phrase of, gold_loc after
    stop-word removal and lemmatizing (as defined in ProPara Section 5).
    """
    if not pred_loc or not gold_loc:
        return False
    gold_norm = normalize_loc(gold_loc)
    pred_norm = normalize_loc(pred_loc)
    if not pred_norm or not gold_norm:
        return False
    return pred_norm <= gold_norm  # subset ≡ sub-phrase


def set_torch_deterministic(seed):
    """Set reproducibility controls for Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_samples(data, step_tokens=None):
    """
    For each paragraph, each step index i, each participant p:
        context  = "step 1: ... step i+1: ..."
        query    = participant, step index, previous state
        answer   = next state after the current step

    states[p][i]   = state of p BEFORE step i+1 (initial state when i=0)
    states[p][i+1] = state of p AFTER  step i+1
    """
    samples = []
    for item in data:
        sentences = item["sentence_texts"]
        participants = item["participants"]
        states = item["states"]

        for step_idx in range(len(sentences)):
            parts = []
            for i in range(step_idx + 1):
                parts.append(f"step {i + 1}: {sentences[i]}")
                if step_tokens and i < len(step_tokens):
                    parts.append(step_tokens[i])
            context = "\n".join(parts)

            for p_idx, participant in enumerate(participants):
                p_states = states[p_idx]
                # states has len(sentences)+1 entries: [initial, after_step1, ...]
                if len(p_states) <= step_idx + 1:
                    continue
                prev_state = map_state(p_states[step_idx])
                curr_state = map_state(p_states[step_idx + 1])
                change_type = compute_state_change(prev_state, curr_state)
                answer = format_answer(curr_state)

                prompt = (
                    f"{context}\n\n"
                    f"participant: {participant}\n"
                    f"step: {step_idx + 1}\n"
                    f"prev: {prev_state}\n"
                    "next:"
                )
                samples.append(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "para_id": item["para_id"],
                        "step_idx": step_idx,
                        "participant": participant,
                        "change_type": change_type,
                        "prev_state": prev_state,
                        "curr_state": curr_state,
                    }
                )
    return samples


class ProParaDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.items = []
        for s in samples:
            full_text = s["prompt"] + " " + s["answer"] + tokenizer.eos_token
            enc = tokenizer(
                full_text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            # Compute prompt length by subtracting the answer token count from the
            # full encoding.  Re-tokenizing "prompt + space" alone is wrong for BPE
            # tokenizers (e.g. GPT-2) because the trailing space merges with the first
            # answer word into a single token, causing that token's loss to be masked.
            answer_ids = tokenizer(
                " " + s["answer"] + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            prompt_len = enc["input_ids"].shape[1] - len(answer_ids)

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


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


def _build_eval_groups(samples, predictions):
    from collections import defaultdict

    groups = defaultdict(list)
    for s, pred in zip(samples, predictions):
        pred_state = parse_prediction(pred)
        pred_change = compute_state_change(s["prev_state"], pred_state)
        key = (s["para_id"], s["participant"])
        groups[key].append((s["step_idx"], s, pred_state, pred_change))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def _collect_change_sets(steps, change_type):
    gold_changes = [
        (step_idx, s)
        for step_idx, s, _, _ in steps
        if s["change_type"] == change_type
    ]
    pred_step_set = {
        step_idx for step_idx, _, _, pred_change in steps if pred_change == change_type
    }
    return gold_changes, pred_step_set


def _find_pred_state_for_step(steps, step_idx):
    for pidx, _, pred_state, _ in steps:
        if pidx == step_idx:
            return pred_state
    return ""


def _score_cat3_for_change_type(steps, gold_changes, change_type):
    correct = 0
    total = 0

    for step_idx, s in gold_changes:
        pred_state = _find_pred_state_for_step(steps, step_idx)
        pred_change = compute_state_change(s["prev_state"], pred_state)

        if change_type == "create":
            gold_loc = s["curr_state"]
            pred_loc = pred_state if pred_change == "create" else ""
            if gold_loc not in NONE_VALS and gold_loc not in UNKNOWN_VALS:
                correct += int(location_match(pred_loc, gold_loc))
                total += 1

        elif change_type == "destroy":
            gold_loc = s["prev_state"]
            pred_loc = s["prev_state"] if pred_change == "destroy" else ""
            if gold_loc not in NONE_VALS and gold_loc not in UNKNOWN_VALS:
                correct += int(location_match(pred_loc, gold_loc))
                total += 1

        elif change_type == "move":
            gold_from = s["prev_state"]
            gold_to = s["curr_state"]
            pred_from = s["prev_state"] if pred_change == "move" else ""
            pred_to = pred_state if pred_change == "move" else ""
            if gold_from not in NONE_VALS and gold_from not in UNKNOWN_VALS:
                correct += int(location_match(pred_from, gold_from))
                total += 1
            if gold_to not in NONE_VALS and gold_to not in UNKNOWN_VALS:
                correct += int(location_match(pred_to, gold_to))
                total += 1

    return correct, total


def evaluate_cat123(samples, predictions):
    """
    Compute the three ProPara evaluation categories (Section 5 of Dalvi et al. 2018).

    Cat-1: Is e created/destroyed/moved in the process?
           Accuracy over all (paragraph, participant, change_type) triples.

    Cat-2: When (step#) is e created/destroyed/moved?
           F1 over predicted vs gold step-index sets.
           Evaluated only for participants where the change actually occurs.

    Cat-3: Where is e created/destroyed/moved from/to?
           Accuracy with partial location match (predicted ⊆ gold after
           stop-word removal and lemmatizing).
           Evaluated only for participants where the change actually occurs.
    """
    groups = _build_eval_groups(samples, predictions)

    cat1_correct = cat1_total = 0
    cat2_tp = cat2_fp = cat2_fn = 0
    cat3_correct = cat3_total = 0

    for (para_id, participant), steps in groups.items():
        for change_type in ("create", "destroy", "move"):
            gold_changes, pred_step_set = _collect_change_sets(steps, change_type)

            # Cat-1: did the change occur at all?
            gold_cat1 = len(gold_changes) > 0
            pred_cat1 = len(pred_step_set) > 0
            cat1_correct += int(gold_cat1 == pred_cat1)
            cat1_total += 1

            if not gold_cat1:
                continue

            # Cat-2: which steps?
            gold_step_set = {idx for idx, _ in gold_changes}
            tp = len(gold_step_set & pred_step_set)
            cat2_tp += tp
            cat2_fp += len(pred_step_set) - tp
            cat2_fn += len(gold_step_set) - tp

            # Cat-3: where?
            add_correct, add_total = _score_cat3_for_change_type(
                steps, gold_changes, change_type
            )
            cat3_correct += add_correct
            cat3_total += add_total

    cat1_acc = cat1_correct / cat1_total if cat1_total else 0.0
    prec = cat2_tp / (cat2_tp + cat2_fp) if (cat2_tp + cat2_fp) > 0 else 0.0
    rec = cat2_tp / (cat2_tp + cat2_fn) if (cat2_tp + cat2_fn) > 0 else 0.0
    cat2_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    cat3_acc = cat3_correct / cat3_total if cat3_total else 0.0

    return {
        "cat1_accuracy": cat1_acc,
        "cat2_f1": cat2_f1,
        "cat3_accuracy": cat3_acc,
        "cat1_total": cat1_total,
        "cat3_total": cat3_total,
    }


def evaluate(
    model,
    tokenizer,
    data,
    device,
    max_length=512,
    max_gen_tokens=64,
    step_tokens=None,
):
    model.eval()
    samples = make_samples(data, step_tokens=step_tokens)
    predictions = []

    with torch.no_grad():
        for s in tqdm(samples, desc="Evaluating"):
            prompt = s["prompt"]
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )["input_ids"].to(device)

            output = model.generate(
                input_ids,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = output[0][input_ids.shape[1] :]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(pred)

    # Per-step change-type accuracy (quick sanity metric)
    gold_types = [s["change_type"] for s in samples]
    pred_states = [parse_prediction(p) for p in predictions]
    pred_types = [
        compute_state_change(s["prev_state"], pstate)
        for s, pstate in zip(samples, pred_states)
    ]
    gold_known = gold_types
    pred_known = pred_types
    step_acc = (
        sum(g == p for g, p in zip(gold_known, pred_known)) / len(gold_known)
        if gold_known
        else 0.0
    )

    cat = evaluate_cat123(samples, predictions)

    return {
        "step_accuracy": step_acc,
        **cat,
        "predictions": predictions,
        "samples": samples,
    }


def main(args):
    set_torch_deterministic(args.seed)

    # Derive output_dir and results_dir dynamically based on model_name
    model_name_path = os.path.normpath(args.model_dir)
    sub_path = ""
    parts = model_name_path.split(os.sep)
    if "models" in parts:
        idx = parts.index("models")
        if len(parts) > idx + 2:
            # Skip 'models' and the dataset name (e.g. 'recipenlg')
            sub_path = os.path.join(*parts[idx+2:])
    else:
        sub_path = model_name_path.replace(os.sep, "_")

    seed_leaf = f"seed={args.seed}"
    if sub_path:
        args.output_dir = os.path.join(args.output_dir, sub_path, seed_leaf)
        args.results_dir = os.path.join(args.results_dir, sub_path, seed_leaf)
    else:
        args.output_dir = os.path.join(args.output_dir, seed_leaf)
        args.results_dir = os.path.join(args.results_dir, seed_leaf)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model_from_checkpoint(args.model_dir, device=device)
    print(f"Model: {args.model_dir}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    step_tokens = None
    if args.use_step_tokens:
        step_tokens = [f"<step_{i}>" for i in range(args.stp_max_steps)]
        existing = set(tokenizer.get_vocab().keys())
        missing = [t for t in step_tokens if t not in existing]
        if missing:
            num_added = tokenizer.add_tokens(missing, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} missing step tokens to tokenizer", flush=True)
        print(f"Using {len(step_tokens)} step tokens: {step_tokens[:3]} ...", flush=True)

    train_data = load_propara_jsonl(args.train_path)
    dev_data = load_propara_jsonl(args.dev_path)
    test_data = load_propara_jsonl(args.test_path)
    if args.perc_samples < 1.0:
        train_data = train_data[:int(len(train_data) * args.perc_samples)]
        dev_data = dev_data[:int(len(dev_data) * args.perc_samples)]
        test_data = test_data[:int(len(test_data) * args.perc_samples)]
    print(
        f"Paragraphs — train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}",
        flush=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.eval_only:
        train_samples = make_samples(train_data, step_tokens=step_tokens)
        print(f"Train samples: {len(train_samples)}", flush=True)

        dataset = ProParaDataset(
            train_samples, tokenizer, max_length=model.config.max_position_embeddings
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )

        optimizer = AdamW(model.parameters(), lr=args.lr)
        model.train()
        global_step = 0
        num_steps = 0

        for epoch in range(args.epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                if args.max_steps > 0 and num_steps >= args.max_steps:
                    break

                if epoch == 0 and num_steps == 0:
                    print("\n=== First Batch Decode (input_ids vs labels) ===", flush=True)
                    n_show = min(8, batch["input_ids"].shape[0])
                    for i in range(n_show):
                        decoded_input = tokenizer.decode(
                            batch["input_ids"][i], skip_special_tokens=False
                        )
                        label_ids = batch["labels"][i]
                        decoded_labels = tokenizer.decode(
                            label_ids[label_ids != -100], skip_special_tokens=False
                        )
                        print(f"Sample {i}", flush=True)
                        print(f"input_ids: {decoded_input}", flush=True)
                        print(f"labels:    {decoded_labels}", flush=True)
                        print("-" * 80, flush=True)

                batch = {k: v.to(device) for k, v in batch.items()}
                # import pdb; pdb.set_trace()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                num_steps += 1

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\nCheckpoint saved: {ckpt_dir}", flush=True)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}", flush=True)

            dev_results = evaluate(
                model,
                tokenizer,
                dev_data,
                device,
                max_length=model.config.max_position_embeddings,
                step_tokens=step_tokens,
            )
            print(
                f"Dev — step_acc: {dev_results['step_accuracy']:.4f} | "
                f"Cat-1: {dev_results['cat1_accuracy']:.4f} | "
                f"Cat-2 F1: {dev_results['cat2_f1']:.4f} | "
                f"Cat-3: {dev_results['cat3_accuracy']:.4f}",
                flush=True,
            )
            model.train()

        final_dir = os.path.join(args.output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Final model saved: {final_dir}", flush=True)

    # Test evaluation
    print("\n=== Test Evaluation ===", flush=True)
    test_results = evaluate(
        model,
        tokenizer,
        test_data,
        device,
        max_length=model.config.max_position_embeddings,
        step_tokens=step_tokens,
    )
    print(
        f"Test — step_acc: {test_results['step_accuracy']:.4f} | "
        f"Cat-1: {test_results['cat1_accuracy']:.4f} | "
        f"Cat-2 F1: {test_results['cat2_f1']:.4f} | "
        f"Cat-3: {test_results['cat3_accuracy']:.4f}",
        flush=True,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "test_results.json")
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(
            {
                "step_accuracy": test_results["step_accuracy"],
                "cat1_accuracy": test_results["cat1_accuracy"],
                "cat2_f1": test_results["cat2_f1"],
                "cat3_accuracy": test_results["cat3_accuracy"],
                "cat1_total": test_results["cat1_total"],
                "cat3_total": test_results["cat3_total"],
                "predictions": [
                    {
                        "prompt": s["prompt"],
                        "pred": p,
                        "gold": s["answer"],
                        "para_id": s["para_id"],
                        "step": s["step_idx"],
                        "participant": s["participant"],
                        "change_type": s["change_type"],
                    }
                    for s, p in zip(
                        test_results["samples"],
                        test_results["predictions"],
                    )
                ],
            },
            f,
            indent=2,
        )
    print(f"Results saved: {results_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2-medium")
    parser.add_argument(
        "--train_path",
        default="./data/propara/data/emnlp18/grids.v1.train.json",
    )
    parser.add_argument(
        "--dev_path",
        default="./data/propara/data/emnlp18/grids.v1.dev.json",
    )
    parser.add_argument(
        "--test_path",
        default="./data/propara/data/emnlp18/grids.v1.test.json",
    )
    parser.add_argument("--perc_samples", default=1.0, type=float)
    parser.add_argument("--output_dir", default="./models/propara")
    parser.add_argument("--results_dir", default="./results/propara")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--save_interval", default=0, type=int)
    parser.add_argument(
        "--use_step_tokens",
        default=0,
        type=int,
        help="Append <step_i> tokens after each step in the context",
    )
    parser.add_argument(
        "--stp_max_steps",
        default=15,
        type=int,
        help="Number of step tokens available (<step_0> .. <step_N-1>)",
    )
    parser.add_argument("--eval_only", default=0, type=int)
    parser.add_argument(
        "--max_steps",
        default=10000,
        type=int,
        help="Stop training after this many gradient steps (0 = no limit)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for deterministic PyTorch mode",
    )
    args = parser.parse_args()
    main(args)
