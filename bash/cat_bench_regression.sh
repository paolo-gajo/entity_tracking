#!/bin/bash
#SBATCH -J cb-regr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

# BASE_DIR="openai-community/gpt2"
# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"

# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"
# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"

# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=1.0/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"
# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=1.0/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"

# BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=1.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"
BASE_DIR="models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2"

interval=1000

SCRIPT_NAME="${BASH_SOURCE[0]:-$0}"

# Number of models to run in parallel per array task (per GPU)
PARALLEL_PER_TASK=${PARALLEL_PER_TASK:-1}

# 1. SETUP MODE: Count folders and print the sbatch command
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Scanning '$BASE_DIR' for train_config.json files..."

    # Count the number of models found
    mapfile -t configs < <(find "$BASE_DIR" -name "train_config.json")

    model_list_filtered=()
    for config in "${configs[@]}"; do
        train_config_name="${config%/*}"
        num_steps="${train_config_name##*/}"

        if (( num_steps % interval == 0 )); then
            model_list_filtered+=("${config}")
        fi        
    done

    NUM_MODELS=${#model_list_filtered[@]}

    if [ "$NUM_MODELS" -eq 0 ]; then
        echo "Error: No train_config.json files found in $BASE_DIR!"
        exit 1
    fi

    NUM_TASKS=$(( (NUM_MODELS + PARALLEL_PER_TASK - 1) / PARALLEL_PER_TASK ))

    echo "Found $NUM_MODELS models, $PARALLEL_PER_TASK parallel per GPU → $NUM_TASKS array tasks."
    echo "--------------------------------------------------------"
    echo "To submit this job array, run:"
    echo ""
    echo "sbatch --array=1-${NUM_TASKS} $SCRIPT_NAME"
    echo ""
    echo "(Override with: PARALLEL_PER_TASK=8 sbatch --array=1-... $SCRIPT_NAME)"
    return 0 2>/dev/null || exit 0
fi

# 2. EXECUTION MODE: Find all folders, pick this task's chunk, run in parallel
nvidia-smi
module load arrow
source .env/bin/activate

mapfile -t all_models < <(find "$BASE_DIR" -name "train_config.json" -exec dirname {} \; | sort -V)

model_list_filtered=()
for model_dir in "${all_models[@]}"; do
    num_steps="${model_dir##*/}"
    echo $num_steps

    if (( num_steps % interval == 0 )); then
        model_list_filtered+=("${model_dir}")
    fi        
done

START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * PARALLEL_PER_TASK ))
END_IDX=$(( START_IDX + PARALLEL_PER_TASK ))
if [ "$END_IDX" -gt "${#model_list_filtered[@]}" ]; then
    END_IDX=${#model_list_filtered[@]}
fi

echo "Task $SLURM_ARRAY_TASK_ID: running models $((START_IDX+1)) to $END_IDX of ${#model_list_filtered[@]} in parallel"

sample_type="all"
repeat=0

pids=()

for (( i=START_IDX; i<END_IDX; i++ )); do
    model_dir="${model_list_filtered[$i]}"
    echo "Launching [$((i+1))/${#model_list_filtered[@]}]: $model_dir"
    
    python src/cat_bench_regression.py \
            --model_dir "$model_dir" \
            --sample_type "$sample_type" \
            --repeat "$repeat" &
    
    pids+=($!)

done

# Wait for all and report failures
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        ((failed++))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "$failed process(es) failed"
    exit 1
fi
echo "All models completed successfully"