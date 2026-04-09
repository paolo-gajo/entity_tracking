#!/bin/bash
#SBATCH -J run1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

model_name=Qwen/Qwen3-1.7B

python src/cat_bench_thinking_sft.py --model_name $model_name --train 1 --eval 0 --verbose 1 --max_train_steps 10 --num_test_samples 10

# python src/cat_bench_regression.py --step_interval 10000 --model_dir ./models/bw/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-04-03--15-50-04

# python src/cat_bench_regression.py --step_interval 10000 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-16--17-13-15

# python src/sims.py --step_interval 0 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-22--06-37-53
# python src/sims.py --step_interval 0 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-18--04-49-34
# python src/sims.py --step_interval 0 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-21--03-17-44
# python src/sims.py --step_interval 0 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=4/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/Qwen3-0.6B-Base/2026-03-29--16-57-35
