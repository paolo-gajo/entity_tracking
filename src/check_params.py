import os
import json
import torch
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

def get_layer_group(param_name):
    """
    Groups parameters to keep the plot readable.
    Filters out LayerNorms and biases.
    """
    if 'bias' in param_name or 'norm' in param_name or 'ln_' in param_name:
        return None

    if 'wpe' in param_name:
        return 'Positional Embeddings (wpe)'
    if 'wte' in param_name or 'embed_tokens' in param_name:
        return 'Token Embeddings'
    if 'lm_head' in param_name:
        return 'LM Head'

    target_layers = {'0': 'Bottom', '5': 'Middle', '11': 'Top'}
    for layer_num, depth_label in target_layers.items():
        if f'h.{layer_num}.' in param_name or f'layers.{layer_num}.' in param_name:
            if 'attn' in param_name or 'self_attn' in param_name:
                return f'Layer {layer_num} ({depth_label}) Attention'
            if 'mlp' in param_name:
                return f'Layer {layer_num} ({depth_label}) MLP'

    return None

def main(args):
    # 1. Discover and sort model checkpoints
    checkpoints = []
    print(f"Scanning {args.model_dir} for train_config.json...")

    for root, dirs, files in os.walk(args.model_dir):
        if "train_config.json" in files:
            config_path = os.path.join(root, "train_config.json")
            try:
                with open(config_path, "r", encoding="utf8") as f:
                    config = json.load(f)
                    num_steps = config.get("num_steps", 0)
                    checkpoints.append({"path": root, "num_steps": num_steps})
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

    if not checkpoints:
        print("No valid checkpoints found. Exiting.")
        return

    checkpoints = sorted(checkpoints, key=lambda x: x["num_steps"])
    steps = [ckpt["num_steps"] for ckpt in checkpoints]
    print(f"Found {len(checkpoints)} checkpoints. Steps: {steps}")

    # Read train config for display
    cfg_path = os.path.join(checkpoints[0]["path"], "train_config.json")
    with open(cfg_path, "r", encoding="utf8") as f:
        train_config = json.load(f)

    # Get orig vocab size from baseline
    baseline_name = train_config.get("model_name")
    if baseline_name is None:
        print("ERROR: could not detect base model from train_config.json.")
        return
    print(f"Loading baseline to determine vocab size and initial weights: {baseline_name}")
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_name, device_map="cpu")
    orig_vocab_sizes = {}
    W_init = {}
    for name, param in baseline_model.named_parameters():
        clean_name = name.replace("base_model.model.", "")
        is_embedding = ('wte' in clean_name or 'embed_tokens' in clean_name or 'lm_head' in clean_name)
        if is_embedding:
            orig_vocab_sizes[clean_name] = param.shape[0]
            W_init[clean_name + '_orig'] = param.detach().clone()
            W_init[clean_name + '_new'] = torch.zeros(0, param.shape[1])  # placeholder; filled at first ckpt
        else:
            group = get_layer_group(clean_name)
            if group:
                W_init[clean_name] = param.detach().clone()
    del baseline_model

    # 2. Track distances (W_init is now the pretrained baseline)
    trajectories = {}

    for ckpt in tqdm(checkpoints, desc="Processing Checkpoints"):
        ckpt_path = ckpt["path"]
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cpu")

        step_dists = {}
        step_counts = {}

        for name, param in model.named_parameters():
            clean_name = name.replace("base_model.model.", "")

            group = get_layer_group(clean_name)
            if not group:
                continue

            W_t = param.detach()
            is_embedding = ('wte' in clean_name or 'embed_tokens' in clean_name
                            or 'lm_head' in clean_name)

            if is_embedding and (clean_name + '_orig') in W_init:
                orig_len = W_init[clean_name + '_orig'].shape[0]
                W0_orig = W_init[clean_name + '_orig']

                # Capture new token embeddings on first checkpoint they appear
                if W_init[clean_name + '_new'].shape[0] == 0 and W_t.shape[0] > orig_len:
                    W_init[clean_name + '_new'] = W_t[orig_len:].clone()

                W_t_orig = W_t[:orig_len]
                dist_orig = ((W_t_orig - W0_orig) ** 2).mean().sqrt().item()
                g_orig = group + ' (original)'
                step_dists.setdefault(g_orig, 0.0)
                step_counts.setdefault(g_orig, 0)
                step_dists[g_orig] += dist_orig
                step_counts[g_orig] += 1

                if W_t.shape[0] > orig_len and W_init[clean_name + '_new'].shape[0] > 0:
                    W0_new = W_init[clean_name + '_new']
                    W_t_new = W_t[orig_len:]
                    dist_new = ((W_t_new - W0_new) ** 2).mean().sqrt().item()
                    g_new = group + ' (new tokens)'
                    step_dists.setdefault(g_new, 0.0)
                    step_counts.setdefault(g_new, 0)
                    step_dists[g_new] += dist_new
                    step_counts[g_new] += 1

            elif clean_name in W_init:
                W0 = W_init[clean_name]
                diff = W_t - W0
                dist = (diff ** 2).mean().sqrt().item()
                step_dists.setdefault(group, 0.0)
                step_counts.setdefault(group, 0)
                step_dists[group] += dist
                step_counts[group] += 1
                if args.verbose:
                    rms_init = (W0 ** 2).mean().sqrt().item()
                    print(f"  step={ckpt['num_steps']:6d}  {clean_name:<60s}  dist={dist:.4f}  rms_init={rms_init:.4f}")

        del model

        for g in step_dists:
            avg_dist = step_dists[g] / step_counts[g] if step_counts[g] > 0 else 0.0
            trajectories.setdefault(g, []).append(avg_dist)

    # 3. Plotting
    plt.figure(figsize=(14, 8))

    for group, dists in trajectories.items():
        if len(dists) == len(steps):
            plt.plot(steps, dists, marker='o', label=group, alpha=0.8, linewidth=2)
        else:
            print(f"Warning: Length mismatch for '{group}'. Expected {len(steps)}, got {len(dists)}.")

    plt.xlabel("Optimizer Steps", fontsize=12)
    plt.ylabel(r"RMS Displacement $\sqrt{\mathrm{mean}((W_t - W_0)^2)}$", fontsize=12)
    plt.title("Parameter Displacement Trajectories", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize='medium')

    skip_keys = {'num_steps', 'data_path'}
    config_lines = [f"{k}: {v}" for k, v in train_config.items() if k not in skip_keys]
    config_text = "\n".join(config_lines)
    plt.gcf().text(
        1.02, 0.02, config_text,
        transform=plt.gca().transAxes,
        fontsize=7, fontfamily='monospace', verticalalignment='bottom',
    )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(args.model_dir, "displacement_trajectories.pdf")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track L2 displacement of model weights over training steps.")
    parser.add_argument("--model_dir", type=str, required=True, help="Parent directory containing all checkpoint subfolders.")
    parser.add_argument("--verbose", action="store_true", help="Print per-parameter distances for debugging.")
    args = parser.parse_args()
    main(args)