import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from itertools import permutations
import numpy as np
from tqdm.auto import tqdm

def main():    
    model_id = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.eval()

    json_path = './data/recipenlg/recipenlg_samples_10.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    step_list = data[0]['directions']
    perm_list = list(permutations(step_list))
    input_text = ' '.join(step_list)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    loss_list = []
    perplexity_list_original = []
    perplexity_list_shuffled = []
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        loss_original = outputs.loss
        perplexity_list_original.append(torch.exp(loss_original))
        for t in tqdm(perm_list):
            input_text_shuffled = ' '.join(t)
            # print(f'Shuffled: {input_text_shuffled}')
            inputs_shuffled = tokenizer(input_text_shuffled, return_tensors="pt").to(device)
            outputs_shuffled = model(input_ids=inputs_shuffled["input_ids"], labels=inputs_shuffled["input_ids"])
            loss_shuffled = outputs_shuffled.loss
            loss_list.append(loss_shuffled)
            perplexity_list_shuffled.append(torch.exp(loss_shuffled))

    mean_loss_original = sum(loss_list) / len(loss_list)
    perplexity_tensor_original = torch.tensor(perplexity_list_original)
    mean_perplexity_original = sum(perplexity_tensor_original) / len(perplexity_tensor_original)
    min_perplexity_original = min(perplexity_tensor_original)
    max_perplexity_original = max(perplexity_tensor_original)

    print(f"Mean loss original: {mean_loss_original.item():.4f}")
    print(f"Mean perplexity original: {mean_perplexity_original.item():.4f}")
    print(f"Min perplexity original: {min_perplexity_original.item():.4f}")
    print(f"Max perplexity original: {max_perplexity_original.item():.4f}")
    perp_argmin = perplexity_tensor_original.argmin().to(dtype=torch.long)
    print(f"Argmin perplexity original: {perm_list[perp_argmin]}")
    perp_argmax = perplexity_tensor_original.argmax().to(dtype=torch.long)
    print(f"Argmax perplexity original: {perm_list[perp_argmax]}")

    print('#' * 100)

    mean_loss_shuffled = sum(loss_list) / len(loss_list)
    perplexity_tensor_shuffled = torch.tensor(perplexity_list_shuffled)
    mean_perplexity_shuffled = sum(perplexity_tensor_shuffled) / len(perplexity_tensor_shuffled)
    min_perplexity_shuffled = min(perplexity_tensor_shuffled)
    max_perplexity_shuffled = max(perplexity_tensor_shuffled)

    print(f"Mean loss shuffled: {mean_loss_shuffled.item():.4f}")
    print(f"Mean perplexity shuffled: {mean_perplexity_shuffled.item():.4f}")
    print(f"Min perplexity shuffled: {min_perplexity_shuffled.item():.4f}")
    print(f"Max perplexity shuffled: {max_perplexity_shuffled.item():.4f}")
    perp_argmin = perplexity_tensor_shuffled.argmin().to(dtype=torch.long)
    print(f"Argmin perplexity shuffled: {perm_list[perp_argmin]}")
    perp_argmax = perplexity_tensor_shuffled.argmax().to(dtype=torch.long)
    print(f"Argmax perplexity shuffled: {perm_list[perp_argmax]}")

if __name__ == "__main__":
    main()