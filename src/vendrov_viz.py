import matplotlib.pyplot as plt
import json
import numpy as np
import os

RESULTS_DIRS = [
    # "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2",
    # "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium",
    "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-23--17-46-43",
    # "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/SmolLM2-135M",
    "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium",
    "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-22--19-20-21",
    "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-25--17-42-12",
    "./results/erfgc_reachability/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-21--03-17-44",
]

def extract_path_flags(path: str) -> str:
    segments = path.split('/')
    flags = []
    for segment in segments:
        if '=' in segment:
            flags.extend(segment.split('-'))
        elif segment in {'pos_neg', 'random_samples', 'minimal_pairs', 'completion_only'}:
            flags.append(segment)
    return '\n'.join(flags)

def get_argmax_step(y_vals, x_vals) -> str:
    if len(y_vals) == 0 or len(x_vals) == 0:
        return "N/A"
    return str(x_vals[np.argmax(y_vals)])

def smooth(y, window=1):
    y = np.array(y)
    window = min(window, len(y))
    if window < 2:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')

window = 10

N = len(RESULTS_DIRS)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig_dist = plt.figure(figsize=(12, 6))
ax_dist = plt.gca()
fig_shuffled = plt.figure(figsize=(12, 6))
ax_shuffled = plt.gca()
fig_sum = plt.figure(figsize=(12, 6))
ax_sum = plt.gca()

for i, results_dir in enumerate(RESULTS_DIRS):
    data = []
    for root, dirs, files in os.walk(results_dir):
        for F in files:
            if F == "results.json":
                json_path = os.path.join(root, F)
                with open(json_path, 'r', encoding='utf8') as f:
                    data.append(json.load(f))

    model_name = data[-1]['train_config']['model_name'].split('/')[-1]
    prompt_type = data[-1]['train_config']['prompt_type']
    BASELINE_PATH = f'./results/erfgc_reachability/baseline/{model_name}/activations=real/0/results.json'
    # print(BASELINE_PATH)
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, 'r', encoding='utf8') as f:
            baseline_data = json.load(f)
    data = sorted(data, key=lambda x: x.get('train_config', {}).get('num_steps', 0))
    data = [baseline_data] + data
    if not data:
        continue

    auc_directed_list_unshuffled = smooth([el['results']['directed']['unshuffled']['mu'] for el in data])
    y_arr_auc_directed_list_unshuffled = np.array(auc_directed_list_unshuffled)
    y_smooth_auc_directed_list_unshuffled = np.convolve(y_arr_auc_directed_list_unshuffled, np.ones(window)/window, mode='valid')
    auc_directed_list_topological = smooth([el['results']['directed']['topological']['mu'] for el in data])
    y_arr_auc_directed_list_topological = np.array(auc_directed_list_topological)
    y_smooth_auc_directed_list_topological = np.convolve(y_arr_auc_directed_list_topological, np.ones(window)/window, mode='valid')
    auc_directed_list_permutations = smooth([el['results']['directed']['permutations']['mu'] for el in data])
    y_arr_auc_directed_list_permutations = np.array(auc_directed_list_permutations)
    y_smooth_auc_directed_list_permutations = np.convolve(y_arr_auc_directed_list_permutations, np.ones(window)/window, mode='valid')
    auc_undirected_list_unshuffled = smooth([el['results']['undirected']['unshuffled']['mu'] for el in data])
    y_arr_auc_undirected_list_unshuffled = np.array(auc_undirected_list_unshuffled)
    y_smooth_auc_undirected_list_unshuffled = np.convolve(y_arr_auc_undirected_list_unshuffled, np.ones(window)/window, mode='valid')
    auc_undirected_list_topological = smooth([el['results']['undirected']['topological']['mu'] for el in data])
    y_arr_auc_undirected_list_topological = np.array(auc_undirected_list_topological)
    y_smooth_auc_undirected_list_topological = np.convolve(y_arr_auc_undirected_list_topological, np.ones(window)/window, mode='valid')
    auc_undirected_list_permutations = smooth([el['results']['undirected']['permutations']['mu'] for el in data])
    y_arr_auc_undirected_list_permutations = np.array(auc_undirected_list_permutations)
    y_smooth_auc_undirected_list_permutations = np.convolve(y_arr_auc_undirected_list_permutations, np.ones(window)/window, mode='valid')

    x = [el.get('train_config', {}).get('num_steps', 0) for el in data]
    x_smooth = x[(window-1)//2 : (window-1)//2 + len(y_smooth_auc_directed_list_unshuffled)]

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    ax.set_title(f'{model_name} - {prompt_type}', fontsize=10)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
    
    ax.plot(x_smooth, y_smooth_auc_directed_list_unshuffled, label='directed, unshuffled')
    ax.plot(x_smooth, y_smooth_auc_directed_list_topological, label='directed, topological')
    ax.plot(x_smooth, y_smooth_auc_directed_list_permutations, label='directed, permutations')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 1)

    flags_text = extract_path_flags(results_dir)
    argmax_text = (
        "Argmax [Step]:\n"
        f"Dir Unshuf: {get_argmax_step(y_smooth_auc_directed_list_unshuffled, x)}\n"
        f"Dir Topo: {get_argmax_step(y_smooth_auc_directed_list_topological, x)}\n"
        f"Dir Perm: {get_argmax_step(y_smooth_auc_directed_list_permutations, x)}\n"
        f"Undir Unshuf: {get_argmax_step(auc_undirected_list_unshuffled, x)}\n"
        f"Undir Topo: {get_argmax_step(auc_undirected_list_topological, x)}\n"
        f"Undir Perm: {get_argmax_step(auc_undirected_list_permutations, x)}"
    )
    combined_text = f"{flags_text}\n{'='*15}\n{argmax_text}"
    ax.text(
        1.05, 0.45, combined_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='square,pad=1.2', facecolor='white', edgecolor='gray', alpha=0.8)
    )
    if not os.path.exists('./viz'):
        os.makedirs('./viz')
    fig.savefig(f'./viz/fig_{i}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Plot dist onto the shared bottom subplot
    dist = y_smooth_auc_directed_list_unshuffled - y_smooth_auc_directed_list_permutations
    ax_dist.plot(x_smooth, dist, color=colors[i % len(colors)], label=f'model={model_name}, prompt_type={prompt_type}')

    # ax_shuffled.plot(x, y_smooth_auc_directed_list_unshuffled,   color=colors[i % len(colors)], linestyle='-',  label=f'model={model_name}, prompt_type={prompt_type}, order=unshuffled')
    # ax_shuffled.plot(x, y_smooth_auc_directed_list_topological,  color=colors[i % len(colors)], linestyle='--', label=f'model={model_name}, prompt_type={prompt_type}, order=topological')
    ax_shuffled.plot(x_smooth, y_smooth_auc_directed_list_permutations, color=colors[i % len(colors)], linestyle='-',  label=f'model={model_name}, prompt_type={prompt_type}, order=permutations')
    
    normed = abs(y_smooth_auc_directed_list_permutations) - 0.5
    ax_sum.plot(x_smooth, normed, label=f'model={model_name}, {prompt_type}')
    code_str = r"""
    Code:
    normed = abs(y_smooth_auc_directed_list_permutations) - 0.5
    ax_sum.plot(x, normed, label=f'{prompt_type} permutations')
    """
    ax.text(
        1.05, 0.45, code_str, transform=ax_sum.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='square,pad=1.2', facecolor='white', edgecolor='gray', alpha=0.8)
    )

ax_dist.set_title('dist (unshuffled − permutations)', fontsize=11)
ax_dist.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax_dist.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax_dist.set_xlabel('Steps')
ax_dist.set_ylabel('dist')
fig_dist.savefig(f'./viz/fig_dist.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_dist)

ax_shuffled.set_title('Directed-Permutations AUC', fontsize=11)
ax_shuffled.set_ylim(0, 1)
ax_shuffled.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
ax_shuffled.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax_shuffled.set_xlabel('Steps')
ax_shuffled.set_ylabel('AUC')
fig_shuffled.savefig(f'./viz/fig_shuffled.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_shuffled)

ax_sum.set_title('Directed-Permutations AUC (Centered at 0.5)')
ax_sum.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax_sum.axhline(0, color='gray', linewidth=0.8, linestyle='--')
fig_sum.savefig(f'./viz/fig_sum.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_sum)