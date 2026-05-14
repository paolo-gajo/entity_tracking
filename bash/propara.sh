#!/bin/bash
set -euo pipefail

module load arrow
source .env/bin/activate

# model_dir="openai-community/gpt2-medium"
# model_dir="models/recipenlg/mode=random_samples/neg_ratio=0.0/bs=8/prompt=minimal_mono/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-17--03-24-51/100000"
model_dir="models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=minimal_pairs/attn=full/loss=full/clm=1/kl=0/mml=0/pos=0/stp=0/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-18--04-49-34/109000"
# model_dir="models/recipenlg/mode=random_samples/neg_ratio=0.5/bs=8/prompt=step_token_pairs/attn=full/loss=full/clm=0/kl=0/mml=0/pos=0/stp=1/cos=0/eos_init=0/use_lora=0/abs_pe=0/act=real/gpt2-medium/2026-03-23--17-46-43/210000"
seeds=(3 4 5 6 7 8 9)

for seed in "${seeds[@]}"; do
  echo "***** Running ProPara finetuning with seed=${seed} and model_dir=${model_dir} *****"
  python src/finetune_propara.py \
    --model_dir "${model_dir}" \
    --seed "${seed}"
done
