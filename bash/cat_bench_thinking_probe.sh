#!/bin/bash
#SBATCH -J cat_bench_thinking_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

# python src/cat_bench_thinking_probe.py --thinking 0 --sample_frac 0.1
python src/cat_bench_thinking_probe.py --thinking 1 --sample_frac 0.1 --model_dir Qwen/Qwen3-1.7B
python src/cat_bench_thinking_probe.py --thinking 1 --sample_frac 0.1 --model_dir models/recipenlg/mode=grpo/neg_ratio=0.5/bs=1/prompt=grpo_step_tokens/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=1/abs_pe=0/act=real/Qwen3-1.7B/2026-04-05--07-58-50