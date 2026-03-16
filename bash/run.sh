#!/bin/bash
#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log

module load arrow
source .env/bin/activate

# STP
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium
python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-large

# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt-neo-125m
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-small-italian
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/japanese-gpt2-small
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/opt-125m
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/opt-350m
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=completion_only/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/pythia-160m

# CLM
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=completion_only/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-15--19-49-38
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-15--19-49-38
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-15--19-46-28
# python src/cat_bench_regression.py --model_dir models/recipenlg/mode=random_samples/neg_ratio=1.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2/2026-03-15--19-49-37