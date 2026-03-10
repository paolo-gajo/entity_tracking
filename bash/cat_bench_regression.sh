#!/bin/bash
#SBATCH -J cb-regr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

# STP
# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/eos_init=0/use_lora=1/act=real/Qwen3-0.6B-Base"
# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/eos_init=0/use_lora=0/act=real/Qwen3-0.6B-Base"

# MML
# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=1/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/act=non-negative/SmolLM2-135M"
BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=1/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/act=real/SmolLM2-135M"

# COS
# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=0/cos=1/eos_init=0/use_lora=0/act=real/SmolLM2-135M"

# BASELINE
# BASE_DIR="Qwen/Qwen3-0.6B-Base"
# BASE_DIR="HuggingFaceTB/SmolLM2-135M"

# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=1/pos=0/stp=0/eos_init=0/use_lora=0/act=real/gpt2"
# BASE_DIR="models/recipenlg/mode=random_samples/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=1/pos=1/stp=0/eos_init=0/use_lora=0/act=real/gpt2"

SCRIPT_NAME="${BASH_SOURCE[0]:-$0}"

# 1. SETUP MODE: Count folders and print the sbatch command
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Scanning '$BASE_DIR' for train_config.json files..."
    
    # Count the number of models found
    NUM_MODELS=$(find "$BASE_DIR" -name "train_config.json" | wc -l)
    
    if [ "$NUM_MODELS" -eq 0 ]; then
        echo "Error: No train_config.json files found in $BASE_DIR!"
        exit 1
    fi
    
    echo "Found $NUM_MODELS models."
    echo "--------------------------------------------------------"
    echo "To submit this job array, run:"
    echo ""
    echo "sbatch --array=1-${NUM_MODELS} $SCRIPT_NAME"
    echo ""
    return 0 2>/dev/null || exit 0
fi

# 2. EXECUTION MODE: Find the folders again, sort naturally (-V), and pick the Nth one
nvidia-smi
module load arrow
source .env/bin/activate

# Use sort -V so 2000 comes before 10000
model_dir=$(find "$BASE_DIR" -name "train_config.json" -exec dirname {} \; | sort -V | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$model_dir" ]; then
    echo "Error: Could not extract model_dir for task $SLURM_ARRAY_TASK_ID"
    return 1 2>/dev/null || exit 1
fi

echo "Running task ID $SLURM_ARRAY_TASK_ID for model: $model_dir"

sample_type="all"
repeat=0

python src/cat_bench_regression.py \
    --model_dir "$model_dir" \
    --sample_type "$sample_type" \
    --repeat "$repeat"