#!/bin/bash
#SBATCH -J cb-regr-stp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load arrow
source .env/bin/activate
model_dir="models/recipenlg/mode=random_samples/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/act=real/gpt2"
sample_type=all
repeat=0
cmd="python src/cat_bench_regression_step_tokens.py
--model_dir $model_dir
--sample_type $sample_type
--repeat $repeat
"
$cmd
