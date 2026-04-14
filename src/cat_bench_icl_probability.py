import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
from torch.utils.data import DataLoader
from utils_data import ICLDataset, pad_collate
from tqdm.auto import tqdm
import os
import json
import numpy as np
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
import argparse


def main(args):

    model_save_dir = "./results/cat_bench_icl"

    # ----------------------------
    # Load Data
    # ----------------------------
    train_path = "./data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json"
    test_path  = "./data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json"

    df_train = pd.read_json(train_path)
    df_test  = pd.read_json(test_path)

    df_train = df_train[df_train["type"] == "real"].sample(frac=1)
    df_test  = df_test[df_test["type"] == "real"].sample(frac=1)

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

        # ----------------------------
        # Load Model
        # ----------------------------
        model_name = m['path']
        add_prefix_space = "gpt2" in model_name

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=add_prefix_space
        )
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
            print(f"-> Loading model from: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16,
            ).to(device)
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
        # ----------------------------
        # Build Dataset
        # ----------------------------
        max_length = getattr(model.config, "max_position_embeddings", 4096)
        dataset = ICLDataset(
            icl_dataset=df_train,
            test_dataset=df_test,
            tokenizer=tokenizer,
            n_icl=args.n_icl,
            max_length=max_length,
            num_samples=args.num_samples,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: pad_collate(x, tokenizer, side="left"),
        )

        # ----------------------------
        # Yes / No Token IDs
        # ----------------------------
        yes_variants = [" yes", " Yes", "yes", "Yes"]
        no_variants  = [" no", " No", "no", "No"]

        def get_valid_ids(words):
            ids = []
            for w in words:
                token_ids = tokenizer.encode(w, add_special_tokens=False)
                if len(token_ids) == 1:
                    ids.append(token_ids[0])
            return list(set(ids))

        yes_ids = get_valid_ids(yes_variants)
        no_ids  = get_valid_ids(no_variants)

        print("Yes tokens:", tokenizer.convert_ids_to_tokens(yes_ids))
        print("No tokens :", tokenizer.convert_ids_to_tokens(no_ids))

        # ----------------------------
        # Evaluation Loop
        # ----------------------------
        model.eval()

        all_scores    = []
        all_preds     = []
        all_labels    = []
        all_prob_yes  = []
        all_prob_no   = []

        for i, batch in enumerate(tqdm(dataloader)):

            batch = {k: v.to(device) for k, v in batch.items()}

            if i == 0:
                # Save decoded first sample for inspection
                first_ids = batch["input_ids"][0]
                first_attn = batch["attention_mask"][0] == 1
                decoded = tokenizer.decode(first_ids[first_attn], skip_special_tokens=False)
                os.makedirs("./misc", exist_ok=True)
                with open("./misc/cat_bench_icl_sample.txt", "w", encoding="utf-8") as f:
                    f.write(decoded)
                print(f"Saved decoded first sample to ./misc/cat_bench_icl_sample.txt")
            import pdb; pdb.set_trace()
            # Remove last token to predict next token after "Answer:"
            input_ids = batch["input_ids"][:, :-1]
            attention_mask = batch["attention_mask"][:, :-1]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            next_token_logits = outputs.logits[:, -1, :]

            # ---- Hard prediction (prob comparison) ----
            probs = F.softmax(next_token_logits, dim=-1)
            prob_yes = probs[:, yes_ids].sum(dim=1)
            prob_no  = probs[:, no_ids].sum(dim=1)

            preds = (prob_yes > prob_no).long()

            # ---- Continuous log-likelihood ratio ----
            logp = F.log_softmax(next_token_logits, dim=-1)
            score = (
                torch.logsumexp(logp[:, yes_ids], dim=1)
                - torch.logsumexp(logp[:, no_ids], dim=1)
            )

            # Store
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(score.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_prob_yes.extend(prob_yes.cpu().tolist())
            all_prob_no.extend(prob_no.cpu().tolist())

            # Debug first batch
            if i == 0:
                print("Example log-likelihood scores:", score[:5].to(dtype=torch.float16).cpu().numpy())
                print("Prob Yes:", prob_yes[:5].to(dtype=torch.float16).cpu().numpy())
                print("Prob No :", prob_no[:5].to(dtype=torch.float16).cpu().numpy())
                print("------")

        # ----------------------------
        # Metrics
        # ----------------------------
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average = 'macro')
        roc = roc_auc_score(all_labels, all_scores)
        pr  = average_precision_score(all_labels, all_scores)

        print("\nModel:", m['path'])
        print("Accuracy:", acc)
        print("F1:", f1)
        print("ROC AUC:", roc)
        print("PR AUC:", pr)
        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds, digits=4))

        # ----------------------------
        # Save
        # ----------------------------
        os.makedirs(model_save_dir, exist_ok=True)

        # Per-sample predictions
        predictions = []
        for idx in range(len(all_labels)):
            predictions.append({
                "gold_label": all_labels[idx],
                "pred_label": all_preds[idx],
                "score": all_scores[idx],
                "prob_yes": all_prob_yes[idx],
                "prob_no": all_prob_no[idx],
            })

        results = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": pr,
            "predictions": predictions,
        }

        save_path = os.path.join(
            model_save_dir,
            f"results_{'_'.join(m['path'].split('/')[:-2])}.json",
        )

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print("\nSaved to:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2", help="Path or HF name of model")
    parser.add_argument("--n_icl", default=3, type=int, help="Number of ICL examples per group")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of test samples (0 = all)")
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()
    main(args)