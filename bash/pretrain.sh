#!/bin/bash
#SBATCH -J pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load gcc arrow
source .env/bin/activate

data_path='./data/recipenlg/recipenlg_clean.json'
# data_path='./data/recipenlg/recipenlg_clean_100k.json'

model_name="openai-community/gpt2"
# model_name="Qwen/Qwen3-0.6B-Base"
num_samples=0
save_interval=1000
batch_size=16
lr=5e-5
# prompt_type='natlang'
prompt_type='minimal'
# prompt_type='only_shuffled'
# prompt_type='only_original'

# loss_type='full_loss'
loss_type='prompt_only_loss'

use_order_loss=1

cmd="python src/pretrain.py
--data_path $data_path
--model_name $model_name
--prompt_type $prompt_type
--loss_type $loss_type
--num_samples $num_samples
--save_interval $save_interval
--batch_size $batch_size
--lr $lr
--use_order_loss $use_order_loss
"

$cmd
