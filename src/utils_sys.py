from datetime import datetime
import json
import os
import glob as globmod

def setup_config(train_config):
    # Dictionary mapping config keys to short abbreviations for folder names
    abbr = {
        'batch_mode': 'mode',
        'batch_size': 'bs',
        'prompt_type': 'prompt',
        'attn_mask_type': 'attn',
        'clm_mask_type': 'loss',
        'use_clm': 'clm',
        'use_kl': 'kl',
        'use_mml': 'mml',
        'use_grl': 'pos',
        'use_stp': 'stp',
        'use_cos': 'cos',
        'init_from_eos': 'eos_init',
        'use_lora': 'use_lora',
        'use_abs_pe': 'abs_pe',
        'activations': 'act',
    }

    # Keys to include in the directory path (in this specific order)
    grouping_keys = [
        'batch_mode',
        'neg_ratio',
        'batch_size',
        'prompt_type',
        'attn_mask_type',
        'clm_mask_type',
        'use_clm',
        'use_kl',
        'use_mml',
        'use_grl',
        'use_stp',
        'use_cos',
        'init_from_eos',
        'use_lora',
        'use_abs_pe',
        'activations',
    ]
    rev_string = f"_{train_config['revision']}" if train_config['revision'] else ""
    if train_config.get('resume_from'):
        model_leaf = f"{train_config['resume_from'].split('/')[-2]}{rev_string}"
    else:
        model_leaf = f"{train_config['model_name'].split('/')[-1]}{rev_string}"
    
    # Build dynamic string: e.g. "bs=8/prompt=minimal_pairs/..."
    dynamic_subdirs = []
    for k in grouping_keys:
        if k in train_config:
            # Use abbreviation if it exists, otherwise fallback to the full key
            key_name = abbr.get(k, k)
            dynamic_subdirs.append(f"{key_name}={train_config[k]}")
    train_config['prompt_type_list'] = train_config['prompt_type'].split('+')
    train_config['model_save_dir'] = os.path.join(
        './models',
        train_config['data_path'].split('/')[2],
        *dynamic_subdirs,
        model_leaf
    )
    return train_config

def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

def save_model_tokenizer(model, tokenizer, save_config, model_save_dir, filename = 'train_config.json'):
    import torch
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, filename), 'w', encoding='utf8') as f:
        json.dump(save_config, f, ensure_ascii = False, indent = 4, default = str)

    from utils_model import SmolLM2WithAbsPE
    if isinstance(model, SmolLM2WithAbsPE):
        # Save the base HF model and the abs PE weights separately
        model.base_model.save_pretrained(model_save_dir)
        torch.save(
            model.abs_position_embeddings.state_dict(),
            os.path.join(model_save_dir, 'abs_position_embeddings.pt'),
        )
    else:
        model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f'Model saved to: {model_save_dir}', flush=True)

def save_prompt_example(sample_prompt, model_save_dir):
    save_path = os.path.join(model_save_dir, 'prompt.txt')
    with open(save_path, 'w', encoding='utf8') as f:
        f.write(sample_prompt if sample_prompt is not None else "")

def capture_source_snapshot(src_dir=None):
    """Read all .py files from src/ into memory. Call once at launch."""
    if src_dir is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot = {}
    for py_file in globmod.glob(os.path.join(src_dir, '**', '*.py'), recursive=True):
        if '__pycache__' in py_file or os.sep + 'old' + os.sep in py_file:
            continue
        rel_path = os.path.relpath(py_file, src_dir)
        with open(py_file, 'r', encoding='utf8') as f:
            snapshot[rel_path] = f.read()
    return snapshot

def save_source_snapshot(model_save_dir, snapshot):
    """Write a previously captured source snapshot to model_save_dir/source_snapshot/."""
    snapshot_dir = os.path.join(model_save_dir, 'source_snapshot')
    for rel_path, content in snapshot.items():
        dest = os.path.join(snapshot_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'w', encoding='utf8') as f:
            f.write(content)

def save_run(save_config, model_save_dir, model, tokenizer, prompt, source_snapshot=None):
    save_model_tokenizer(model, tokenizer, save_config, model_save_dir)
    save_prompt_example(prompt, model_save_dir)
    if source_snapshot is not None:
        save_source_snapshot(model_save_dir, source_snapshot)