#!/bin/bash
#SBATCH -J pretrain_grpo.py
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow opencv
source .env/bin/activate

python src/pretrain_grpo.py --model_name Qwen/Qwen3-1.7B --config configs/config_tina_mod.json --use_lora 0 --use_vllm 1