from datetime import datetime
import json
import os

def setup_config(train_config):
    train_config['model_save_dir'] = os.path.join('./models',
                                'recipenlg',
                                f"batch_size={train_config['batch_size']}",
                                train_config['batch_mode'],
                                train_config['prompt_type'],
                                train_config['attention_mask_type'],
                                (f"clm={train_config['use_causal_lm_loss']}"
                                f"-kl={train_config['use_kl']}"
                                # f"-ol={train_config['use_order_loss']}"
                                f"-mml={train_config['use_max_margin_loss']}"),
                                train_config['model_name'].split('/')[-1],
                                f"activations={train_config['activations']}",
    )
    return train_config

def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

def save_model_tokenizer(model, tokenizer, save_config, model_save_dir, filename = 'train_config.json'):
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, filename), 'w', encoding='utf8') as f:
        json.dump(save_config, f, ensure_ascii = False, indent = 4, default = str)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

def save_prompt_example(sample_prompt, model_save_dir):
    save_path = os.path.join(model_save_dir, 'prompt.txt')
    with open(save_path, 'w', encoding='utf8') as f:
        f.write(sample_prompt if sample_prompt is not None else "")

def save_run(save_config, model_save_dir, model, tokenizer, prompt):
    save_model_tokenizer(model, tokenizer, save_config, model_save_dir)
    save_prompt_example(prompt, model_save_dir)