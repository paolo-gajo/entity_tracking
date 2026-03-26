#!/bin/bash
#SBATCH -J run_cat_bench_regression
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

python src/cat_bench_regression.py --model_dir results/cat_bench_regression/samples=all/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/Qwen3-0.6B-Base/2026-03-24--17-31-16

# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-16--17-13-15
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-21--03-17-44
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs+minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-22--18-22-43
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs+minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-22--18-22-43
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-20--05-03-28
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-21--02-57-01
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=1.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large/2026-03-20--05-03-28