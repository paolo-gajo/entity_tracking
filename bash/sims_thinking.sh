#!/bin/bash
#SBATCH -J sims_thinking
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

# model_dir : can_think (1 = supports <think> trace, 0 = plain instruct)
declare -A models=(
    [Qwen/Qwen3-0.6B]=1
    [Qwen/Qwen3-1.7B]=1
    [Qwen/Qwen3-4B]=1
    [Qwen/Qwen3-8B]=1
    [Qwen/Qwen3-14B]=1
    [Qwen/Qwen3-32B]=1
    # [meta-llama/Llama-3.2-1B-Instruct]=0
    # [meta-llama/Llama-3.2-3B-Instruct]=0
    # [meta-llama/Llama-3.1-8B-Instruct]=0
    # [meta-llama/Llama-3.3-70B-Instruct]=0
    # [mistralai/Mistral-7B-Instruct-v0.3]=0
    # [mistralai/Ministral-8B-Instruct-2410]=0
    # [mistralai/Mistral-Small-24B-Instruct-2501]=0
    # [google/gemma-2-2b-it]=0
    # [google/gemma-2-9b-it]=0
    # [google/gemma-2-27b-it]=0
    # [google/gemma-3-4b-it]=0
    # [google/gemma-3-12b-it]=0
    # [google/gemma-3-27b-it]=0
    # [allenai/OLMo-2-1124-7B-Instruct]=0
    # [allenai/OLMo-2-1124-13B-Instruct]=0
    # [microsoft/Phi-4]=0
    # [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]=1
    # [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]=1
)

repeat=0
thinking=0

for model_dir in "${!models[@]}"; do
    can_think=${models[$model_dir]}
    python src/sims_thinking.py \
        --model_dir "$model_dir" \
        --thinking "$thinking" \
        --can_think "$can_think" \
        --repeat "$repeat"
done