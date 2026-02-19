#!/bin/bash
#SBATCH -J pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

module load arrow
source .env/bin/activate

data_path='./data/recipenlg/recipenlg_clean.json'
# data_path='./data/recipenlg/recipenlg_clean_100k.json'

model_name="openai-community/gpt2"
# model_name="Qwen/Qwen3-0.6B-Base"

num_samples=0
save_interval=1000

lr=5e-5

prompt_type=minimal_pairs
# prompt_type=natlang_pairs
# prompt_type=only_shuffled
# prompt_type=only_original
# prompt_type=minimal_mono

# attention_mask_type='full_input'
attention_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original

batch_mode="random_samples"
batch_size=32

# batch_mode="pos_neg"
# batch_size=1

use_causal_lm_loss=1
use_kl=0
use_order_loss=0
use_max_margin_loss=0

# activations=real
activations=non-negative

cmd="python src/pretrain.py
--data_path $data_path
--model_name $model_name
--prompt_type $prompt_type
--attention_mask_type $attention_mask_type
--num_samples $num_samples
--save_interval $save_interval
--batch_size $batch_size
--lr $lr
--batch_mode $batch_mode
--use_causal_lm_loss $use_causal_lm_loss
--use_kl $use_kl
--use_order_loss $use_order_loss
--use_max_margin_loss $use_max_margin_loss
--activations $activations
"

$cmd
