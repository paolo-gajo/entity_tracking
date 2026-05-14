"""
CaT-Bench probing with thinking prompts.

Prompts a model (e.g. Qwen3-1.7B) with recipe steps and a dependency question,
asks it to output the two steps verbatim.  In --thinking mode the model first
reasons inside <think>...</think> tags before echoing the steps; in baseline
mode (--thinking 0) it echoes them directly.

Embeddings are extracted from the echoed step tokens via a forward pass on
the full (prompt + generated) sequence, then fed to a logistic regression probe.

Usage:
  python src/cat_thinking.py --model_dir Qwen/Qwen3-1.7B
  python src/cat_thinking.py --model_dir Qwen/Qwen3-1.7B --thinking 0
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils.utils_model import load_model_from_checkpoint
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import os
import json
import argparse
import random


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(row, tokenizer, thinking=True):
    """Build a chat prompt that asks the model to echo two steps.

    Returns (prompt_text, step_a_text, step_b_text, idx_a, idx_b).
    """
    steps = list(row["steps"])
    idx_a, idx_b = row["step_pair_idx_asked_about"]
    step_a_text = steps[idx_a]
    step_b_text = steps[idx_b]

    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    user_content = (
        f"Recipe: {row['title']}\n\n"
        f"Steps:\n{steps_text}\n\n"
        f"{row['binary_question']}\n\n"
        f"After your analysis, repeat the two steps being asked about "
        f"exactly as written above, in the format:\n"
        f"Step {idx_a+1}: <exact text>\n"
        f"Step {idx_b+1}: <exact text>"
    )

    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )

    return prompt_text, step_a_text, step_b_text, idx_a, idx_b


# ---------------------------------------------------------------------------
# Token-span finder
# ---------------------------------------------------------------------------

def find_token_span(decoded_tokens, target_text):
    """Find the token span [start, end) whose decoded text contains target_text.

    decoded_tokens: list[str], one per token.
    Returns (start_idx, end_idx) or None.
    """
    cum_text = ""
    char_to_tok = []
    for t_idx, tok_str in enumerate(decoded_tokens):
        for _ in tok_str:
            char_to_tok.append(t_idx)
        cum_text += tok_str

    pos = cum_text.find(target_text)
    if pos == -1:
        return None

    start_tok = char_to_tok[pos]
    end_char = pos + len(target_text) - 1
    if end_char >= len(char_to_tok):
        end_tok = len(decoded_tokens)
    else:
        end_tok = char_to_tok[end_char] + 1

    return start_tok, end_tok


def _extract_step_emb(gen_decoded, step_text, marker, hidden, hidden_dim, device):
    """Locate *step_text* after *marker* in decoded generated tokens and return
    mean-pooled hidden state embedding."""
    fallback = torch.zeros(hidden_dim, device=device)

    marker_span = find_token_span(gen_decoded, marker)
    if marker_span is None:
        return fallback

    after_marker_decoded = gen_decoded[marker_span[1]:]
    span = find_token_span(after_marker_decoded, step_text)
    if span is None:
        # Fallback: use tokens from marker end to next newline
        start = marker_span[1]
        end = start
        cum = ""
        for j in range(start, len(gen_decoded)):
            cum += gen_decoded[j]
            if "\n" in cum:
                end = j
                break
        else:
            end = len(gen_decoded)
        if end <= start:
            return fallback
    else:
        start = marker_span[1] + span[0]
        end = marker_span[1] + span[1]

    end = min(end, hidden.shape[0])
    if start >= end:
        return fallback

    return hidden[start:end].mean(dim=0)


# ---------------------------------------------------------------------------
# Baseline: construct response, single forward pass (no generation)
# ---------------------------------------------------------------------------

def get_step_embeddings_baseline(batch_df, tokenizer, model, device):
    """Construct prompt + known response (just the echoed steps), run a single
    forward pass, and extract mean-pooled embeddings for the step tokens."""
    batch_df = batch_df.reset_index(drop=True)

    full_texts = []
    response_texts = []
    step_a_texts = []
    step_b_texts = []
    idx_a_list = []
    idx_b_list = []

    for _, row in batch_df.iterrows():
        prompt, step_a, step_b, idx_a, idx_b = build_prompt(
            row, tokenizer, thinking=False
        )
        # Construct the known response: just the two steps echoed
        response = f"Step {idx_a+1}: {step_a}\nStep {idx_b+1}: {step_b}"
        full_texts.append(prompt + response)
        response_texts.append(response)
        step_a_texts.append(step_a)
        step_b_texts.append(step_b)
        idx_a_list.append(idx_a)
        idx_b_list.append(idx_b)

    # Tokenize and pad (right-padding, standard forward pass)
    encoded = tokenizer(
        full_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
    ).to(device)

    with torch.no_grad():
        out = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
    hidden = out.hidden_states[-1]  # (batch, seq_len, hidden_dim)

    features = []
    hidden_dim = hidden.shape[-1]

    for i in range(len(batch_df)):
        # Find where the response starts by tokenizing prompt alone
        prompt_text = full_texts[i][: -len(response_texts[i])]
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        # Decode response tokens for span finding
        resp_ids = encoded["input_ids"][i, prompt_len:]
        # Mask out padding
        resp_attn = encoded["attention_mask"][i, prompt_len:]
        resp_len = resp_attn.sum().item()
        resp_ids = resp_ids[:resp_len]

        resp_hidden = hidden[i, prompt_len : prompt_len + resp_len]

        resp_decoded = [
            tokenizer.decode(resp_ids[t : t + 1], skip_special_tokens=False)
            for t in range(len(resp_ids))
        ]

        step_a_marker = f"Step {idx_a_list[i]+1}: "
        step_b_marker = f"Step {idx_b_list[i]+1}: "

        emb_a = _extract_step_emb(
            resp_decoded, step_a_texts[i], step_a_marker,
            resp_hidden, hidden_dim, device,
        )
        emb_b = _extract_step_emb(
            resp_decoded, step_b_texts[i], step_b_marker,
            resp_hidden, hidden_dim, device,
        )

        feat = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=0)
        features.append(feat.cpu().float().numpy())

    return np.array(features), response_texts


# ---------------------------------------------------------------------------
# Thinking: generate, then forward pass
# ---------------------------------------------------------------------------

def get_step_embeddings_thinking(batch_df, tokenizer, model, device,
                                 max_new_tokens=1024):
    """Generate responses (with thinking), run a forward pass on each full
    sequence, and extract mean-pooled embeddings for the echoed step tokens."""
    batch_df = batch_df.reset_index(drop=True)

    prompts = []
    step_a_texts = []
    step_b_texts = []
    idx_a_list = []
    idx_b_list = []

    for _, row in batch_df.iterrows():
        prompt, step_a, step_b, idx_a, idx_b = build_prompt(
            row, tokenizer, thinking=True
        )
        prompts.append(prompt)
        step_a_texts.append(step_a)
        step_b_texts.append(step_b)
        idx_a_list.append(idx_a)
        idx_b_list.append(idx_b)

    # --- 1. Generate responses ---
    tokenizer.padding_side = "left"
    gen_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        gen_sequences = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )

    prompt_len = gen_inputs["input_ids"].shape[1]
    tokenizer.padding_side = "right"

    # --- 2. For each sample: forward pass on full sequence, extract step embeddings ---
    features = []
    generated_texts = []
    hidden_dim = model.config.hidden_size

    for i in range(len(batch_df)):
        # Get non-padded prompt tokens
        attn = gen_inputs["attention_mask"][i]
        prompt_ids = gen_inputs["input_ids"][i][attn == 1]

        # Get generated tokens (strip trailing EOS/pad)
        gen_ids = gen_sequences[i, prompt_len:]
        if tokenizer.eos_token_id is not None:
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0]]

        # Full sequence
        full_ids = torch.cat([prompt_ids, gen_ids], dim=0)
        model_max = getattr(model.config, "max_position_embeddings", 4096)
        full_ids = full_ids[:model_max]

        # Forward pass
        with torch.no_grad():
            out = model(
                input_ids=full_ids.unsqueeze(0).to(device),
                output_hidden_states=True,
            )
        h = out.hidden_states[-1][0]  # (seq_len, hidden_dim)

        # Hidden states for the generated portion only
        gen_start = len(prompt_ids)
        gen_hidden = h[gen_start:]

        # Decode each generated token for span finding
        gen_decoded = [
            tokenizer.decode(gen_ids[t : t + 1], skip_special_tokens=False)
            for t in range(len(gen_ids))
        ]

        step_a_marker = f"Step {idx_a_list[i]+1}: "
        step_b_marker = f"Step {idx_b_list[i]+1}: "

        emb_a = _extract_step_emb(
            gen_decoded, step_a_texts[i], step_a_marker,
            gen_hidden, hidden_dim, device,
        )
        emb_b = _extract_step_emb(
            gen_decoded, step_b_texts[i], step_b_marker,
            gen_hidden, hidden_dim, device,
        )

        feat = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=0)
        features.append(feat.cpu().float().numpy())

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        generated_texts.append(gen_text)

    return np.array(features), generated_texts


# ---------------------------------------------------------------------------
# Data loading + feature extraction
# ---------------------------------------------------------------------------

def load_and_extract(path, tokenizer, model, device, sample_type="real",
                     batch_size=4, sample_limit=None, sample_frac=1.0,
                     thinking=True, max_new_tokens=1024):
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if sample_type != "all":
        df = df[df["type"] == sample_type]

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    if sample_limit:
        df = df.head(sample_limit)

    print(f"Extracting features for {path} ({len(df)} samples)...")

    all_feats = []
    all_gen_texts = []
    all_labels = df["label"].values

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        if thinking:
            feats, gen_texts = get_step_embeddings_thinking(
                batch, tokenizer, model, device,
                max_new_tokens=max_new_tokens,
            )
        else:
            feats, gen_texts = get_step_embeddings_baseline(
                batch, tokenizer, model, device,
            )
        if len(feats) > 0:
            all_feats.append(feats)
            all_gen_texts.extend(gen_texts)

    if not all_feats:
        return np.array([]), np.array([]), []

    return np.concatenate(all_feats, axis=0), all_labels, all_gen_texts


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def get_model_info(model_path, args, task_name="cat_thinking"):
    train_conf_path = os.path.join(model_path, "train_config.json")
    sample_dir = f"samples={args.sample_type}"
    results_base = args.results_base_dir

    if os.path.exists(train_conf_path):
        with open(train_conf_path, "r", encoding="utf8") as f:
            train_config = json.load(f)
        rel = os.path.relpath(model_path, start=os.path.normpath("./models"))
        save_path = os.path.join(results_base, task_name, sample_dir, rel)
        print(f"save_path: {save_path}")
        return save_path, train_config

    train_config = {"num_steps": 0}
    model_leaf = os.path.basename(os.path.normpath(model_path))
    save_path = os.path.join(results_base, task_name, sample_dir, "baseline", model_leaf, "0")
    return save_path, train_config


def save_results_to_disk(results, save_path, train_config, args):
    os.makedirs(save_path, exist_ok=True)
    out_dict = {
        "train_config": train_config,
        "eval_config": vars(args),
        "results": results,
    }
    json_path = os.path.join(save_path, "results.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    data_path_train = "./data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json"
    data_path_test = "./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Walk model_dir for checkpoints or treat as single model
    if not os.path.exists(args.model_dir):
        model_list = [{"path": args.model_dir, "num_steps": 0}]
    else:
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for F in files:
                if F == "train_config.json":
                    with open(os.path.join(root, F), "r", encoding="utf8") as f:
                        num_steps = json.load(f)["num_steps"]
                    model_list.append({"path": root, "num_steps": num_steps})
        if not model_list:
            model_list = [{"path": args.model_dir, "num_steps": 0}]
        model_list = sorted(model_list, key=lambda x: x["num_steps"])

    if args.step_interval > 0:
        model_list = [m for m in model_list if m["num_steps"] % args.step_interval == 0]

    for m in model_list:
        model_name = m["path"]
        task_name = "cat_bench_thinking_probe" if args.thinking else "cat_bench_thinking_probe_baseline"
        save_path, train_config = get_model_info(model_name, args, task_name=task_name)
        result_file = os.path.join(save_path, "results.json")

        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        print(f"Loading model from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        adapter_config_path = os.path.join(model_name, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            print("-> Detected LoRA adapter. Using PEFT two-stage loading...")
            config = PeftConfig.from_pretrained(model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                dtype=torch.bfloat16,
            ).to(device)

            from safetensors import safe_open
            adapter_safetensors = os.path.join(model_name, "adapter_model.safetensors")
            checkpoint_vocab_size = None
            with safe_open(adapter_safetensors, framework="pt") as f:
                for key in f.keys():
                    if "modules_to_save" in key and ("embed_tokens" in key or "lm_head" in key):
                        checkpoint_vocab_size = f.get_slice(key).get_shape()[0]
                        break
            if checkpoint_vocab_size is not None and checkpoint_vocab_size != base_model.config.vocab_size:
                base_model.resize_token_embeddings(checkpoint_vocab_size)

            model = PeftModel.from_pretrained(base_model, model_name)
        else:
            print("-> Loading model...")
            model = load_model_from_checkpoint(model_name, device=device, dtype=torch.bfloat16)
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))

        model.eval()

        # Extract features
        print("--- Processing Train Data ---")
        X_train, y_train, gen_texts_train = load_and_extract(
            data_path_train, tokenizer, model, device,
            sample_type=args.sample_type, sample_frac=args.sample_frac,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            thinking=args.thinking,
        )
        print("--- Processing Test Data ---")
        X_test, y_test, gen_texts_test = load_and_extract(
            data_path_test, tokenizer, model, device,
            sample_type=args.sample_type, sample_frac=args.sample_frac,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            thinking=args.thinking,
        )

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        if X_train.size == 0 or X_test.size == 0:
            print("No features extracted, skipping")
            continue

        # Train logistic regression probe
        print("Training logistic regression probe...")
        clf = LogisticRegression(max_iter=args.max_iter, C=1.0, solver="lbfgs", verbose=1)
        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        results = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "acc": float(np.mean(preds == y_test)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
            "f1_binary": float(f1_score(y_test, preds, average="binary")),
            "y_test": y_test.tolist(),
            "preds": preds.tolist(),
        }
        if len(np.unique(y_test)) == 2:
            results["roc_auc"] = float(roc_auc_score(y_test, probs))

        report = classification_report(y_test, preds, digits=4)

        if args.verbose_results:
            print(f"Model: {model_name}")
            print(report)
            print(f"Macro F1: {results['f1_macro']:.4f}")
            if "roc_auc" in results:
                print(f"ROC AUC: {results['roc_auc']:.4f}")

        results["report"] = report

        # Per-sample predictions (test set)
        predictions = []
        for idx in range(len(y_test)):
            predictions.append({
                "gold_label": int(y_test[idx]),
                "pred_label": int(preds[idx]),
                "pred_prob": float(probs[idx]),
                "generated_text": gen_texts_test[idx] if idx < len(gen_texts_test) else "",
            })
        results["predictions"] = predictions

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CaT-Bench probing with thinking prompts"
    )
    parser.add_argument("--model_dir", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=0, type=int)
    parser.add_argument("--sample_type", default="all", choices=["real", "all"])
    parser.add_argument("--sample_frac", default=1.0, type=float)
    parser.add_argument("--max_iter", default=2000, type=int)
    parser.add_argument("--step_interval", default=0, type=int)
    parser.add_argument("--results_base_dir", default="./results", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=4096, type=int,
                        help="Max tokens to generate (includes thinking + step echoes)")
    parser.add_argument("--thinking", default=1, type=int,
                        help="1 = model thinks in <think> tags before echoing steps; "
                             "0 = baseline, model echoes steps directly")
    args = parser.parse_args()
    main(args)
