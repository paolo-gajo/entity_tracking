import argparse
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def format_pizza_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        title = item.get("title", "")
        # Add a title at the top
        text_seq = f"Recipe: {title}\n"
        for step in item.get("table", []):
            instruction = step.get("instructions", "NA")
            inp = step.get("input", "NA")
            act = step.get("action", "NA")
            out = step.get("output", "NA")
            text_seq += f"Instruction: {instruction} | Input: {inp} | Action: {act} | Output: {out}\n"
        
        texts.append(text_seq)
    
    return Dataset.from_dict({"text": texts})

def main():
    parser = argparse.ArgumentParser(description="Train/Test on Pizza Common Sense Dataset")
    parser.add_argument("--train_file", type=str, default="data/pizza_common_sense/train.json")
    parser.add_argument("--val_file", type=str, default="data/pizza_common_sense/val.json")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="models/pizza_common_sense_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    # Load data
    train_dataset = format_pizza_data(args.train_file)
    val_dataset = format_pizza_data(args.val_file)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Evaluating loss...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    print("Running generated samples (testing)...")
    # For testing generation, we'll take a few validation samples, prompt the model, and print the output
    model.eval()
    device = model.device
    test_samples = val_dataset.select(range(min(3, len(val_dataset))))
    for i, sample in enumerate(test_samples):
        # Taking just the first half of the string as a prompt
        text_content = sample["text"]
        prompt = text_content[:len(text_content)//2]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            top_p=0.9, 
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n--- Test Sample {i+1} ---")
        print(f"PROMPT:\n{prompt}")
        print(f"GENERATED:\n{generated_text}")
        print("-" * 30)

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()