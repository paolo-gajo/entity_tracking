# src/reachability/eval_reachability.py
import json
import argparse
import os

from reachability.utils_reachability import (
    load_data,
    get_model_info,
    process_model,
    save_results_to_disk,
)

def main(args):
    json_files = [f'./data/erfgc/bio/{split}.json' for split in ['train', 'val', 'test']]
    data = load_data(json_files)

    if not os.path.exists(args.model_dir):
        model_list = [{'path': args.model_dir, 'num_steps': 0}]
    else:
        model_list = []
        for root, dirs, files in os.walk(args.model_dir):
            for F in files:
                if F == 'train_config.json':
                    with open(os.path.join(root, F), 'r', encoding='utf8') as f:
                        num_steps = json.load(f)['num_steps']
                    model_list.append({'path': root, 'num_steps': num_steps})
        model_list = sorted(model_list, key=lambda x: x['num_steps'])
        assert len(model_list) == len(set([el['num_steps'] for el in model_list])), "You're os.walking through 2+ model dir trees at once."

    if args.step_interval > 0:
        model_list = [m for m in model_list if m['num_steps'] % args.step_interval == 0]

    for model in model_list:
        model_name = model['path']
        save_path, train_config = get_model_info(model_name, args)
        result_file = os.path.join(save_path, "results.json")
        if os.path.exists(result_file) and not args.repeat:
            print(f"Skipping {model_name}: results exist at {result_file}")
            continue

        print('Processing model...')
        results = process_model(model_name, args, data)

        if args.verbose_results:
            print(json.dumps(results, indent=4))

        if args.save_results:
            save_results_to_disk(results, save_path, train_config, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="openai-community/gpt2")
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--verbose_results", default=1, type=int)
    parser.add_argument("--repeat", default=0, type=int)
    parser.add_argument("--activations", default="real", type=str, help="real | non-negative")
    parser.add_argument("--save_heatmaps", default=1, type=int)
    parser.add_argument("--use_gold_transpose", default=0, type=int)
    parser.add_argument("--step_interval", default=10000, type=int,
                        help="If > 0, only evaluate checkpoints whose num_steps is a multiple of this value")

    args = parser.parse_args()
    main(args)