import matplotlib.pyplot as plt
import json
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

k_formatter = FuncFormatter(lambda x, _: '0' if x == 0 else f'{int(x/1000)}k')

def get_data_from_dir(dir_path):
    data = []
    for root, dirs, files in os.walk(dir_path):
        for F in files:
            if F == "results.json":
                json_path = os.path.join(root, F)
                with open(json_path, 'r', encoding='utf8') as f:
                    data.append(json.load(f))
    return data

MODEL_DIRS = [
    "./results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/japanese-gpt2-small",
    "./results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-small-italian",
    "./results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-small-arabic",
    "./results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-base-french",
    "./results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/german-gpt2",
]

MODEL_LABELS = {
    'japanese-gpt2-small': 'Japanese GPT-2',
    'gpt2-small-italian':   'Italian GPT-2',
    'gpt2-small-arabic':    'Arabic GPT-2',
    'gpt2-base-french':     'French GPT-2',
    'german-gpt2':          'German GPT-2',
}

interval = 10000

plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True, sharex=True)

for ax, model_dir in zip(axes, MODEL_DIRS):
    model_name = model_dir.rstrip('/').split('/')[-1]
    data = get_data_from_dir(model_dir)
    if not data:
        print(f"No data found in {model_dir}")
        continue

    baseline_path = f'./results/cat_bench_regression/samples=all/baseline/{model_name}/0/results.json'
    with open(baseline_path, 'r', encoding='utf8') as f:
        baseline = json.load(f)

    data = sorted([baseline] + data, key=lambda x: x['train_config']['num_steps'])
    data = [el for el in data if el['train_config']['num_steps'] % interval == 0]
    data = [el for el in data if el['train_config']['num_steps'] <= 240000]

    f1_macro = [el['results']['f1_macro'] for el in data]
    x = [el['train_config']['num_steps'] for el in data]

    (line,) = ax.plot(x, f1_macro, linewidth=2.5)
    ax.axhline(baseline['results']['f1_macro'], color='black', ls='--', linewidth=1.5)

    max_val = float(np.max(f1_macro))
    mini_handles = [Line2D([0], [0], color='none'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=line.get_color(), markersize=8)]
    mini_leg = ax.legend(mini_handles, ['Max', f'{max_val:.4f}'], fontsize=20, framealpha=0.0,
                         loc='lower center', bbox_to_anchor=(0.5, 0.75),
                         ncol=2, handletextpad=0, columnspacing=1, handlelength=1)
    ax.add_artist(mini_leg)

    ax.set_title(MODEL_LABELS[model_name], fontsize=20, pad=16)
    ax.tick_params(axis='both', labelsize=20)
    ax.xaxis.set_major_formatter(k_formatter)
    ax.grid(False)

axes[0].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('./paper/probing_results_non_english.pdf', format='pdf', bbox_inches='tight')
plt.show()
