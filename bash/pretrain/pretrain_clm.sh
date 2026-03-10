#!/bin/bash
#SBATCH -J pt-clm-full_loss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load arrow
source .env/bin/activate

save_interval=1000
lr=5e-5

min_recipe_steps=1
neg_ratio=0.5

data_path='./data/recipenlg/recipenlg_clean.json'
num_samples=10000
batch_mode="random_samples"
batch_size=8

# data_path='./data/recipenlg/recipenlg_clean_100k.json'
# num_samples=0
# batch_mode="pos_neg"
# batch_size=1

# model_name="openai-community/gpt2"
# model_name="Qwen/Qwen3-0.6B-Base"
# model_name="facebook/opt-350m"
model_name="EleutherAI/gpt-neo-125m"

attn_mask_type='full' # N/A for minimal_mono, only_shuffled, only_original
# attn_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original

# loss_mask_type='full' # N/A for minimal_mono, only_shuffled, only_original
loss_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original
prompt_type=minimal_pairs
# prompt_type=natlang_pairs

# prompt_type=only_shuffled
# prompt_type=only_original
# prompt_type=minimal_mono

use_clm=1
pool_clm=0
use_kl=0
use_mml=0
use_stp=0

activations=real
# activations=non-negative

cmd="python src/pretrain.py
--data_path $data_path
--model_name $model_name
--prompt_type $prompt_type
--attn_mask_type $attn_mask_type
--loss_mask_type $loss_mask_type
--num_samples $num_samples
--save_interval $save_interval
--batch_size $batch_size
--lr $lr
--batch_mode $batch_mode
--use_clm $use_clm
--pool_clm $pool_clm
--use_kl $use_kl
--use_mml $use_mml
--activations $activations
--min_recipe_steps $min_recipe_steps
--neg_ratio $neg_ratio
--use_stp $use_stp
"

$cmd
