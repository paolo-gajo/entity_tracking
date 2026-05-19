python src/pretrain_grpo.py --model_name models/cat_bench_thinking/Qwen_Qwen3-8B/2026-04-03--23-42-38/final

after doing grpo on recipenlg repeat the probing experiment with src/cat_bench_regression.py, the exact same way, to check if the embeddings are better after this unshuffling training (i guess gotta do causal LM ablation as well)