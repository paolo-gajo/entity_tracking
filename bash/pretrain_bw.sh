#!/bin/bash
#SBATCH -J run_cat_bench_regression
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

data_path="./data/bw/bw_stp_dataset_meta.json"

batch_mode=random_samples

attn_mask_type='full' # N/A for minimal_mono
# attn_mask_type='completion_only' # N/A for minimal_mono, only_shuffled, only_original

clm_mask_type='full' # for minimal_mono, only_shuffled, only_original
# clm_mask_type='completion_only' # for minimal_pairs, step_token_pairs

prompt_type=minimal_pairs
# prompt_type=minimal_mono

use_stp=0
use_clm=1

cmd="python src/pretrain_bw.py
--data_path $data_path
--batch_mode $batch_mode
--use_stp $use_stp
--use_clm $use_clm
--max_train_steps 30000
--attn_mask_type $attn_mask_type
--clm_mask_type $clm_mask_type
--prompt_type $prompt_type
"