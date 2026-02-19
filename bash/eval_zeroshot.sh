#!/bin/bash
#SBATCH -J sims
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load arrow
source .env/bin/activate

bigger_objective=future
non_neg_activations=0

for (( i=1; i<=7; i++ ))
do
    # model_dir="models/recipenlg/pos_neg/minimal_pairs/completion_only/clm=1-kl=0-ol=0-mml=0/gpt2/${i}000"
    # model_dir="models/recipenlg/pos_neg/minimal_pairs/completion_only/clm=0-kl=0-ol=0-mml=1/gpt2/${i}000"
    # model_dir="models/recipenlg/pos_neg/minimal_mono/full_input/clm=0-kl=0-ol=0-mml=1/gpt2/${i}000"
    # model_dir="models/recipenlg/random_samples/minimal_pairs/completion_only/clm=1-kl=0-ol=0-mml=0/gpt2/${i}000"
    # model_dir="models/recipenlg/pos_neg/minimal_pairs/completion_only/clm=1-kl=1-ol=0-mml=1/gpt2/${i}000"
    model_dir="openai-community/gpt2"
    cmd="python src/eval_zeroshot.py
    --bigger_objective $bigger_objective
    --model_dir $model_dir
    --non_neg_activations $non_neg_activations
    "
    $cmd
done
