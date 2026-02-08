#!/bin/bash
#SBATCH -J sims
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load gcc arrow
source .env/bin/activate

n_runs=10

model_dir=openai-community/gpt2
cmd="python src/sims.py
--n_runs $n_runs
--model_dir $model_dir
"
$cmd