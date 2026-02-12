import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from utils_data import ICLDataset, pad_collate
from tqdm.auto import tqdm
import os
import json
from sklearn.metrics import f1_score, classification_report
import argparse

def main(args):

    model_save_dir = './models/cat_bench'

    data_path_train = './data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json'
    df_train = pd.read_json(data_path_train)
    data_path_test = './data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json'
    df_test = pd.read_json(data_path_test)

    model_name = args.model_dir

    # model_name = "openai-community/gpt2"
    model_name = "models/recipenlg/minimal/prompt_only_loss_with_order_loss/gpt2/1000"
    # model_name = "Qwen/Qwen3-14B-Base"
    # model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-0.6B-Base"
    # model_name = "meta-llama/Llama-3.1-8B"

    add_prefix_space = True if 'gpt2' in model_name else False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=add_prefix_space)

    max_length = 2048

    if 'gpt2' in model_name:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
        max_length = 1024
    if model_name in ["meta-llama/Llama-3.1-8B"]:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    n_icl = 1
    df_test = df_test[df_test['type'] == 'real']
    df_test = df_test.sample(frac=1)
    num_samples = 0
    dataset = ICLDataset(icl_dataset=df_train,
                        test_dataset=df_test,
                        tokenizer=tokenizer,
                        n_icl=n_icl,
                        max_length=max_length,
                        num_samples=num_samples,
                        )
    batch_size = 8

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = lambda x: pad_collate(x, tokenizer, side = 'left'))

    yes_variants = [" yes", " Yes", "yes", "Yes"]
    no_variants = [" no", " No", "no", "No"]

    # Helper to get IDs and filter out invalid ones
    def get_valid_ids(tokenizer, words):
        ids = []
        for w in words:
            # Check if token exists in vocab
            token_id = tokenizer.encode(w, add_special_tokens=False)
            # Only accept if it maps to a single token
            if len(token_id) == 1:
                ids.append(token_id[0])
        return list(set(ids))

    yes_token_ids = get_valid_ids(tokenizer, yes_variants)
    no_token_ids = get_valid_ids(tokenizer, no_variants)

    print(f"Tracking 'Yes' variants: {tokenizer.convert_ids_to_tokens(yes_token_ids)} {yes_token_ids}")
    print(f"Tracking 'No' variants: {tokenizer.convert_ids_to_tokens(no_token_ids)} {no_token_ids}")

    # Gold label map (Mapping the single token used in dataset construction to 0/1)
    # We assume the dataset constructed labels using specifically " yes" and " no"
    ds_yes_id = tokenizer(" yes", add_special_tokens = False)['input_ids'][0]
    ds_no_id = tokenizer(" no", add_special_tokens = False)['input_ids'][0]

    # 2. Corrected Loop
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda x: pad_collate(x, tokenizer, side='left'))

    tbar = tqdm(dataloader)
    preds = []
    golds = []

    model.eval()

    for i, batch in enumerate(tbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Slice Input vs Target
        input_ids = batch['input_ids'][:, :-1]
        attention_mask = batch['attention_mask'][:, :-1]
        batch_golds = batch['input_ids'][:, -1]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        next_token_logits = outputs.logits[:, -1, :]
        
        # --- ROBUST DECODING ---
        # Sum probabilities of all "Yes" variants vs all "No" variants
        # Note: We work in probability space to sum correctly, then go back to class
        probs = F.softmax(next_token_logits, dim=-1)
        
        prob_yes = probs[:, yes_token_ids].sum(dim=1)
        prob_no = probs[:, no_token_ids].sum(dim=1)
        
        # Class 0 = No, Class 1 = Yes
        batch_preds = (prob_yes > prob_no).long().tolist()
        
        # --- GOLD EXTRACTION ---
        batch_golds_binary = []
        for token_id in batch_golds:
            if token_id == ds_yes_id:
                batch_golds_binary.append(1)
            elif token_id == ds_no_id:
                batch_golds_binary.append(0)
            else:
                batch_golds_binary.append(-1) # Ignore or flag

        preds.extend(batch_preds)
        golds.extend(batch_golds_binary)
        
        # --- DEBUGGING (First batch only) ---
        if i == 0:
            top_tokens = torch.argmax(next_token_logits, dim=-1)
            print("Top predicted token:", tokenizer.batch_decode(top_tokens))
            print(f"Prob Yes: {prob_yes.cpu().numpy()}")
            print(f"Prob No:  {prob_no.cpu().numpy()}")
            print("--------------------------------------\n")

    # Calculate F1
    f1 = f1_score(golds, preds, average='macro') # Use macro or binary as needed
    report = classification_report(golds, preds)
    print(model_name)
    print(report)
    print(f"F1 Score: {f1}")

    report_json_path = os.path.join(model_save_dir, f"results_{model_name.split('/')[-1]}.json")
    # Ensure directory exists
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)

    with open(report_json_path, 'w', encoding='utf8') as f:
        json.dump({'f1': f1}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval an LLM on CaT-Bench through ICL")
    parser.add_argument("--model_dir", help="model dir")
    args = parser.parse_args()
    main(args)