import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from utils_data import make_cat_bench_sample, pad_collate
from utils_models import prep_inputs_for_causal_lm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import json

data_path = './data/cat_bench/catplan-data-release/generated_questions/train_must_why/train_must_why.json'
model_save_dir = './models/cat_bench'

df = pd.read_json(data_path)

labels_nl = ('no', 'yes')

model_name = "openai-community/gpt2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

if model_name == 'openai-community/gpt2':
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id

dataset = df.apply(lambda x: make_cat_bench_sample(x, tokenizer, train=True), axis = 1)

batch_size = 16

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = lambda x: pad_collate(x, tokenizer))

loss_fn = CrossEntropyLoss()
lr = 5e-5
optimizer = AdamW(params = model.parameters(), lr = lr)
tbar = tqdm(dataloader)
losses = []
for batch in tbar:
    batch = prep_inputs_for_causal_lm(
        labels=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        bos_token_id=tokenizer.bos_token_id,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch['labels']
    labels[labels == tokenizer.pad_token_id] = -100
    
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits
    optimizer.zero_grad()
    loss = loss_fn(logits.view(-1, model.config.vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()
    tbar.set_description(desc = str(loss.item()))
    losses.append(loss.item())

model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

losses_json_path = os.path.join(model_save_dir, 'losses.json')

with open(losses_json_path, 'w', encoding='utf8') as f:
    json.dump(losses, f, ensure_ascii = False, indent = 4)