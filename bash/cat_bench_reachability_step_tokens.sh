#!/bin/bash
#SBATCH -J cbr-stp-all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log
#SBATCH --array=0-N

module load arrow
source .env/bin/activate

save_results=1
repeat=1
sample_type=all
stp_max_steps=15

# Define model directories
model_dirs=(
"openai-community/gpt2"
"models/recipenlg/mode=random_samples/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/act=real/gpt2"
)

n_models=${#model_dirs[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    
    model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID]}

    if [[ "$model_dir" == *"activations=non-negative"* ]]; then
        activations=non-negative
    else
        activations=real
    fi

    cmd=(python src/cat_bench_reachability_step_tokens.py
    --model_dir "$model_dir"
    --sample_type "$sample_type"
    --activations "$activations"
    --save_results "$save_results"
    --repeat "$repeat"
    --stp_max_steps "$stp_max_steps"
    )

    printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
elif [[ $1 ]]; then
    for (( i=0; i<${#model_dirs[@]}; i++ ))
    do
        model_dir=${model_dirs[$i]}

        if [[ "$model_dir" == *"activations=non-negative"* ]]; then
            activations=non-negative
        else
            activations=real
        fi

        cmd=(python src/cat_bench_reachability_step_tokens.py
        --model_dir "$model_dir"
        --sample_type "$sample_type"
        --activations "$activations"
        --save_results "$save_results"
        --repeat "$repeat"
        --stp_max_steps "$stp_max_steps"
        )

        printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
        "${cmd[@]}"
    done
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((n_models-1)) ${BASH_SOURCE[0]}"
    echo "This will distribute $n_models jobs across N GPUs."
fi
