#!/bin/bash
#SBATCH -J run2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

# python src/cat_bench_regression.py --step_interval 10000 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs+minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-22--18-22-43

# python src/sims.py --step_interval 10000 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-22--19-20-21
# python src/sims.py --step_interval 10000 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-23--17-46-43
python src/sims.py --step_interval 10000 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-25--17-42-12
# python src/sims.py --step_interval 10000 --n_runs 1 --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=4/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/Qwen3-0.6B-Base/2026-03-29--20-04-04