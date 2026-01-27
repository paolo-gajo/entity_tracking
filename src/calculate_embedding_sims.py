from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from utils_data import ListOfDictsDataset
import json

json_path = './data/erfgc/bio/test.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

dataset = ListOfDictsDataset(data)

model_name = 'openai-community/gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

import pdb; pdb.set_trace()