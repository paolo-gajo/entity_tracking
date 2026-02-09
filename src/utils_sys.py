from datetime import datetime
import os
import json

def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

def save_model_tokenizer(model, tokenizer, train_config):
    model_save_dir = os.path.join('./models/recipenlg',
                                  train_config['prompt_type'],
                                  train_config['loss_type'],
                                  train_config['model_name'].split('/')[-1],
                                  str(train_config['steps']),
                                  )
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, 'train_config.json'), 'w', encoding='utf8') as f:
        json.dump(train_config, f, ensure_ascii = False, indent = 4)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
