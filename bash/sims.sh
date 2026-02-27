#!/bin/bash
#SBATCH -J sims
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

n_runs=1
save_results=1
repeat=1
use_gold_transpose=0

# Define model directories
model_dirs=(
# "openai-community/gpt2"
# "models/recipenlg/random_samples/batch_size=8/minimal_pairs/attn_mask_type=full/loss_mask_type=completion_only/clm=0-kl=0-mml=1/no_pos_mml=0/gpt2/activations=real"
# "models/recipenlg/random_samples/batch_size=8/minimal_pairs/attn_mask_type=full/loss_mask_type=completion_only/clm=0-kl=0-mml=1/no_pos_mml=1/gpt2/activations=real"
"models/recipenlg/random_samples/batch_size=8/minimal_pairs/attn_mask_type=full/loss_mask_type=completion_only/clm=0-kl=0-mml=1/use_pos_adv=1/gpt2/activations=real"
)

n_models=${#model_dirs[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    
    model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID]}

    if [[ "$model_dir" == *"activations=non-negative"* ]]; then
        activations=non-negative
    else
        activations=real
    fi

    cmd=(python src/sims_reachability.py
    --n_runs "$n_runs"
    --model_dir "$model_dir"
    --save_results "$save_results"
    --repeat "$repeat"
    --activations "$activations"
    --use_gold_transpose $use_gold_transpose
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

        cmd=(python src/sims.py
        --n_runs "$n_runs"
        --model_dir "$model_dir"
        --save_results "$save_results"
        --repeat "$repeat"
        --activations "$activations"
        --use_gold_transpose $use_gold_transpose
        )

        printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
        "${cmd[@]}"
    done
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((n_models-1)) ${BASH_SOURCE[0]}"
    echo "This will distribute $n_models jobs across N GPUs."
fi
