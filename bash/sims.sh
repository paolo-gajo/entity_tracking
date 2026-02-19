#!/bin/bash
#SBATCH -J sims
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log
#SBATCH --array=0-N

module load arrow
source .env/bin/activate

n_runs=1
save_results=1
use_transitive_closure=0
repeat=0

# Define model directories
model_dirs=(
"models/recipenlg/batch_size=8/random_samples/minimal_pairs/completion_only/clm=1-kl=0-mml=0/gpt2/activations=real"
"models/recipenlg/batch_size=8/random_samples/minimal_pairs/completion_only/clm=1-kl=0-mml=0/gpt2/activations=non-negative"
)

n_models=${#model_dirs[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    
    model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID]}

    if [[ "$model_dir" == *"activations=non-negative"* ]]; then
        activations=non-negative
    else
        activations=real
    fi

    cmd=(python src/sims.py
    --n_runs "$n_runs"
    --model_dir "$model_dir"
    --save_results "$save_results"
    --use_transitive_closure "$use_transitive_closure"
    --repeat "$repeat"
    --activations "$activations"
    )

    printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"

else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((n_models-1)) ${BASH_SOURCE[0]}"
    echo "This will distribute $n_models jobs across N GPUs."
fi
