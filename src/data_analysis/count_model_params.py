import json
from transformers import AutoModelForCausalLM

MODELS = [
    {
        "model_id": "dbmdz/german-gpt2",
        "hf_url": "https://huggingface.co/dbmdz/german-gpt2",
    },
    {
        "model_id": "ClassCat/gpt2-base-french",
        "hf_url": "https://huggingface.co/ClassCat/gpt2-base-french",
    },
    {
        "model_id": "akhooli/gpt2-small-arabic",
        "hf_url": "https://huggingface.co/akhooli/gpt2-small-arabic",
    },
    {
        "model_id": "GroNLP/gpt2-small-italian",
        "hf_url": "https://huggingface.co/GroNLP/gpt2-small-italian",
    },
    {
        "model_id": "rinna/japanese-gpt2-small",
        "hf_url": "https://huggingface.co/rinna/japanese-gpt2-small",
    },
]

OUTPUT_PATH = "model_params.json"

results = []
for entry in MODELS:
    model_id = entry["model_id"]
    print(f"Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    total_params = sum(p.numel() for p in model.parameters())
    del model
    results.append({
        "model_id": model_id,
        "hf_url": entry["hf_url"],
        "num_params": total_params,
    })
    print(f"  {model_id}: {total_params:,} params")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
