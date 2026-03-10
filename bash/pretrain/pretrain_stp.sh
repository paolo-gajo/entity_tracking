#!/bin/bash
#SBATCH -J pt-stp
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
num_samples=1000000
batch_mode="random_samples"
batch_size=8

# data_path='./data/recipenlg/recipenlg_clean_100k.json'
# num_samples=0
# batch_mode="pos_neg"
# batch_size=1

# model_name="openai-community/gpt2"
# model_name="openai-community/gpt2-medium"
model_name="openai-community/gpt2-large"
# model_name="EleutherAI/gpt-neo-125m"
# model_name="Qwen/Qwen3-0.6B-Base"
# model_name="Qwen/Qwen3-4B-Base"
# model_name="Qwen/Qwen3.5-0.8B-Base"
# model_name="Qwen/Qwen3.5-9B-Base"
# model_name="Qwen/Qwen3.5-4B-Base"
# model_name="facebook/opt-350m"
# model_name="HuggingFaceTB/SmolLM2-135M"
# model_name="HuggingFaceTB/SmolLM2-360M"

use_lora=0
prepend_bos=0
detect_anomaly=0

attn_mask_type='full' # N/A for minimal_mono, only_shuffled, only_original
# attn_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original

# loss_mask_type='full' # N/A for minimal_mono, only_shuffled, only_original
loss_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original
prompt_type=step_token_pairs
# prompt_type=natlang_pairs

# prompt_type=only_shuffled
# prompt_type=only_original
# prompt_type=minimal_mono

use_clm=0
use_kl=0
use_mml=0
mml_lambda=0.1

use_stp=1
stp_lambda=1.0
stp_max_steps=15
init_from_eos=0

activations=real
# activations=non-negative

cmd="python src/pretrain.py
--data_path $data_path
--model_name $model_name
--use_lora $use_lora
--prompt_type $prompt_type
--attn_mask_type $attn_mask_type
--num_samples $num_samples
--save_interval $save_interval
--batch_size $batch_size
--lr $lr
--batch_mode $batch_mode
--use_clm $use_clm
--use_kl $use_kl
--use_mml $use_mml
--mml_lambda $mml_lambda
--activations $activations
--min_recipe_steps $min_recipe_steps
--neg_ratio $neg_ratio
--use_stp $use_stp
--stp_lambda $stp_lambda
--stp_max_steps $stp_max_steps
--init_from_eos $init_from_eos
--prepend_bos $prepend_bos
--detect_anomaly $detect_anomaly
"

$cmd
