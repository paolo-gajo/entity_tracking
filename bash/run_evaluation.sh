#!/bin/bash
#SBATCH -J eval_stat_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

RESULTS_DIR="./results_stat_test"
STEP_INTERVAL=10000
COMMON_ARGS="--step_interval $STEP_INTERVAL --repeat 0 --results_base_dir $RESULTS_DIR"

# -------------------------
# 1. Baselines
# -------------------------
echo "=== Running baselines ==="

# python src/cat_bench_regression.py --model_dir openai-community/gpt2 $COMMON_ARGS
# python src/cat_bench_regression.py --model_dir openai-community/gpt2-medium $COMMON_ARGS
# python src/cat_bench_regression.py --model_dir openai-community/gpt2-large $COMMON_ARGS
# python src/cat_bench_regression.py --model_dir Qwen/Qwen3-0.6B-Base $COMMON_ARGS

# -------------------------
# 2. Trained models
# -------------------------
echo "=== Running trained models ==="

MODEL_DIRS=(
    # gpt2
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-22--19-20-21"
    "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-22--06-37-53"

    # gpt2-medium
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-23--17-46-43"
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-18--04-49-34"

    # gpt2-large
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-25--17-42-12"
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-21--03-17-44"

    # Qwen3-0.6B-Base
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=4/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/Qwen3-0.6B-Base/2026-03-29--20-04-04"
    # "models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=4/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/Qwen3-0.6B-Base/2026-03-29--16-57-35"
)

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    echo "--- Evaluating: $MODEL_DIR ---"
    python src/cat_bench_regression.py --model_dir "$MODEL_DIR" $COMMON_ARGS
done

# -------------------------
# 3. Bootstrap statistical test
# -------------------------
echo "=== Running bootstrap statistical test ==="

python src/bootstrap_test.py --results_dir "$RESULTS_DIR" --n_bootstrap 2000

echo "=== Done ==="
