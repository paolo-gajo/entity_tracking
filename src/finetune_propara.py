import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_model import load_model_from_checkpoint
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import argparse
import json
import os

STATE_MAP = {"-": "none", "?": "unknown"}


def load_propara_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def map_state(state):
    return STATE_MAP.get(state, state)


def make_samples(data, step_tokens=None):
    """
    For each paragraph, each step index i, each participant p:
      context  = "Step 1: ... Step 2: ... What is the state of {p}?"
      answer   = states[p][i]          (mapped through STATE_MAP)

    step i means the model has seen sentences 0..i.

    If step_tokens is provided (list of token strings like ["<step_0>", ...]),
    each step's text is followed by its step token:
      "Step 1: ... <step_0> Step 2: ... <step_1> What is the state of {p}?"
    """
    samples = []
    for item in data:
        sentences = item["sentence_texts"]
        participants = item["participants"]
        states = item["states"]

        for step_idx in range(len(sentences)):
            parts = []
            for i in range(step_idx + 1):
                parts.append(f"Step {i + 1}: {sentences[i]}")
                if step_tokens and i < len(step_tokens):
                    parts.append(step_tokens[i])
            context = " ".join(parts)
            for p_idx, participant in enumerate(participants):
                state = map_state(states[p_idx][step_idx])
                prompt = f"{context} What is the state of {participant}?"
                samples.append(
                    {
                        "prompt": prompt,
                        "answer": state,
                        "para_id": item["para_id"],
                        "step_idx": step_idx,
                        "participant": participant,
                    }
                )
    return samples


class ProParaDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.items = []
        for s in samples:
            full_text = s["prompt"] + " " + s["answer"] + tokenizer.eos_token
            enc = tokenizer(
                full_text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            prompt_enc = tokenizer(
                s["prompt"] + " ",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

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


def evaluate(model, tokenizer, data, device, max_length=512, max_gen_tokens=32, step_tokens=None):
    model.eval()
    samples = make_samples(data, step_tokens=step_tokens)
    predictions = []
    gold_labels = []

    with torch.no_grad():
        for s in tqdm(samples, desc="Evaluating"):
            prompt = s["prompt"] + " "
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
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
            gold_labels.append(s["answer"])

    acc = accuracy_score(gold_labels, predictions)
    f1_macro = f1_score(gold_labels, predictions, average="macro", zero_division=0)
    f1_micro = f1_score(gold_labels, predictions, average="micro", zero_division=0)
    report = classification_report(gold_labels, predictions, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "report": report,
        "predictions": predictions,
        "gold": gold_labels,
        "samples": samples,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model_from_checkpoint(args.model_name, device=device)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    step_tokens = None
    if args.use_step_tokens:
        step_tokens = [f"<step_{i}>" for i in range(args.stp_max_steps)]
        # Ensure step tokens exist in tokenizer (they should if loaded from
        # a checkpoint that was pretrained with step tokens)
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
    print(
        f"Paragraphs — train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}",
        flush=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.eval_only:
        train_samples = make_samples(train_data, step_tokens=step_tokens)
        print(f"Train samples: {len(train_samples)}", flush=True)

        dataset = ProParaDataset(train_samples, tokenizer, max_length=model.config.max_position_embeddings)
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
                batch = {k: v.to(device) for k, v in batch.items()}
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
                model, tokenizer, dev_data, device, max_length=model.config.max_position_embeddings,
                step_tokens=step_tokens,
            )
            print(
                f"Dev — acc: {dev_results['accuracy']:.4f}, "
                f"F1 macro: {dev_results['f1_macro']:.4f}, "
                f"F1 micro: {dev_results['f1_micro']:.4f}",
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
        model, tokenizer, test_data, device, max_length=model.config.max_position_embeddings,
        step_tokens=step_tokens,
    )
    print(
        f"Test — acc: {test_results['accuracy']:.4f}, "
        f"F1 macro: {test_results['f1_macro']:.4f}, "
        f"F1 micro: {test_results['f1_micro']:.4f}",
        flush=True,
    )
    print(f"\n{test_results['report']}", flush=True)

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(
            {
                "accuracy": test_results["accuracy"],
                "f1_macro": test_results["f1_macro"],
                "f1_micro": test_results["f1_micro"],
                "predictions": [
                    {
                        "prompt": s["prompt"],
                        "pred": p,
                        "gold": g,
                        "para_id": s["para_id"],
                        "step": s["step_idx"],
                        "participant": s["participant"],
                    }
                    for s, p, g in zip(
                        test_results["samples"],
                        test_results["predictions"],
                        test_results["gold"],
                    )
                ],
            },
            f,
            indent=2,
        )
    print(f"Results saved: {results_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai-community/gpt2")
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
    parser.add_argument("--output_dir", default="./models/propara")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--save_interval", default=0, type=int)
    parser.add_argument("--use_step_tokens", default=0, type=int,
                        help="Append <step_i> tokens after each step in the context")
    parser.add_argument("--stp_max_steps", default=15, type=int,
                        help="Number of step tokens available (<step_0> .. <step_N-1>)")
    parser.add_argument("--eval_only", default=0, type=int)
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="Stop training after this many gradient steps (0 = no limit)")
    args = parser.parse_args()
    main(args)
