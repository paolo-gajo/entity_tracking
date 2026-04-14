#!/bin/bash
#SBATCH -J pretrain_grpo.py
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow opencv
source .env/bin/activate

python src/pretrain_grpo.py --model_name Qwen/Qwen3-1.7B --config configs/config_lora.json --use_lora 1 --use_vllm 0
