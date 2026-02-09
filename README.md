Pre-training:
- ablation: pre-train on just the unshuffled recipes
- ablation: pre-train on just the shuffled recipes

ideas:
- thinking vs non-thinking w/ pretraining vs non-thinking w/o pretraining or anything --> similarity evaluation of embeddings
- plot with topological orders on x axis, pick one model, e.g. the one at 120k steps
- block world: try to see if you can train a model to emit parameters for the creation of block world problems

Unshuffling experiment:

We use two datasets, RecipeNLG and CaT-Bench. RecipeNLG consists of 2.2M samples, where each sample contains a list of strings, which are steps of a culinary recipe. CaT-Bench has a train set of 20,802 samples and a test set of 4260 samples. Each sample in CaT-Bench has a list of steps of recipes, like RecipeNLG, but also has questions e.g. "Must Step 10 happen after Step 8?" where the model needs to answer a binary question about the causal dependence of two of the steps.

I am pre-training a GPT2 model on prompts such as:

```
Below is a jumbled list of recipe steps. Put them in the correct order.

Input:
- In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.
- Let stand until firm, about 30 minutes.
- Stir in vanilla and cereal; mix well.
- Stir over medium heat until mixture bubbles all over top.
- Using 2 teaspoons, drop and shape into 30 clusters on wax paper.
- Boil and stir 5 minutes more. Take off heat.

Correct order:
1. In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.
2. Stir over medium heat until mixture bubbles all over top.
3. Boil and stir 5 minutes more. Take off heat.
4. Stir in vanilla and cereal; mix well.
5. Using 2 teaspoons, drop and shape into 30 clusters on wax paper.
6. Let stand until firm, about 30 minutes.<|endoftext|>
```

We want the model to learn to produce different representations for each step, based on the underlying graph. Using sims.py we calculate the similarity between the step representations and compare them to the adjacency matrix A by calculating the auc(S, A) between A and the step similarity matrix S. When we feed shuffled recipes the AUC is ~0.5, but with unshuffled recipes and recipes shuffled with valid topological orders the AUC is 0.65-0.67. This suggests that the model internally learns the step topology of the recipes.

We want this skill to transfer to the CaT-Bench benchmark:

```
[
    {
        "plan_idx": 0,
        "title": "spicy-tomato-anchovy-pasta",
        "question_idx": 0,
        "steps": [
            "Heat 6 tablespoons olive oil in a large frying pan over medium heat, then stir in garlic, broccoli and mushrooms;",
            "cook until lightly browned.",
            "Add anchovies and water, cover and simmer for 4 to 5 minutes.",
            "Stir in spring onions, tomatoes and parsley and cover again, simmering until vegetables are soft, about 3 to 4 minutes.",
            "While the vegetables are cooking, bring a large pot of water and one teaspoon of olive oil to the boil.",
            "Add linguine and cook until al dente, about 7 to 8 minutes;",
            "drain.",
            "Toss with anchovy mixture and chilli flakes.",
            "If desired, season with black pepper.",
            "Serve immediately."
        ],
        "question_type": "dependent_real_after",
        "step_pair_idx_asked_about": [
            7,
            9
        ],
        "binary_question": "Must Step 10 happen after Step 8?",
        "why_question": "Why must Step 10 happen after Step 8?",
        "label": 1,
        "type": "real",
        "direction": "after"
    }
]
```

where recipes are assigned binary questions on the dependence of steps. Note that CaT-Bench uses data from English Flowgraph Recipe Corpus by Yamakata et al. 2020. RecipeNLG contains some of these, but we have already filtered matches so there is no overlap between pre-training and evaluation.

Currently, the evaluation is done with the script below, but the problem is that the mass for the softmax of the next-token model logits after pre-training is almost all on the eos token. How do we make the model pre-training be useful for the CaT-Bench benchmark?

```
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

model_save_dir = './models/cat_bench'

data_path_train = './data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json'
df_train = pd.read_json(data_path_train)
data_path_test = './data/cat_bench/catplan-data-release/generated_questions/test_must_why/test_must_why.json'
df_test = pd.read_json(data_path_test)

# model_name = "openai-community/gpt2"
# model_name = "models_tested/recipenlg/natlang/full_loss/gpt2_117000"
model_name = "models_tested/recipenlg/natlang/prompt_only_loss/gpt2_48000"
# model_name = "models_tested/recipenlg/minimal/prompt_only_loss/gpt2_91000"
# model_name = "Qwen/Qwen3-14B-Base"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-0.6B-Base"
# model_name = "meta-llama/Llama-3.1-8B"

add_prefix_space = True if model_name == "openai-community/gpt2" else False
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
```

