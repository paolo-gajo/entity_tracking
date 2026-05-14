"""
Paired bootstrap test comparing trained model checkpoints against their baseline.

Reads results.json files (which must contain y_test and preds) from a results directory,
runs paired bootstrap resampling to estimate the F1 delta and its significance.
"""

import numpy as np
import json
import os
import argparse
from sklearn.metrics import f1_score
from glob import glob


def bootstrap_f1_delta(y_true, preds_model, preds_baseline, metric='f1_macro', B=2000, seed=42):
    """
    Paired bootstrap test: is model's F1 significantly > baseline's F1?
    Returns: mean delta, 95% CI, p-value (fraction of bootstrap deltas <= 0).
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    avg = 'macro' if metric == 'f1_macro' else 'binary'
    deltas = []

    for _ in range(B):
        idx = rng.choice(N, size=N, replace=True)
        f1_m = f1_score(y_true[idx], preds_model[idx], average=avg)
        f1_b = f1_score(y_true[idx], preds_baseline[idx], average=avg)
        deltas.append(f1_m - f1_b)

    deltas = np.array(deltas)
    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)
    p_value = np.mean(deltas <= 0)

    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "significant_at_0.05": bool(ci_low > 0),
        "B": B,
    }


def find_baseline_model_name(model_dir_path):
    """
    Extract the base model name (e.g. gpt2-medium) from the trained model path.
    The convention is .../base_model_name/timestamp/checkpoint/
    """
    # Walk up from the model dir to find the base model name
    # Typical path: models/recipenlg/.../gpt2-medium/2026-03-18--04-49-34
    parts = model_dir_path.replace("\\", "/").split("/")
    # The base model name is the second-to-last component (before the timestamp)
    for i, part in enumerate(parts):
        # Timestamps look like 2026-03-18--04-49-34
        if len(part) >= 19 and part[4] == '-' and '--' in part:
            return parts[i - 1]
    # Fallback: look for known model names
    known = ['gpt2', 'gpt2-medium', 'gpt2-large', 'Qwen3-0.6B-Base']
    for name in known:
        if name in model_dir_path:
            return name
    return None


def load_results(results_json_path):
    """Load a results.json and return y_test, preds, and metadata."""
    with open(results_json_path, 'r') as f:
        data = json.load(f)
    results = data['results']
    if 'y_test' not in results or 'preds' not in results:
        return None
    return {
        'y_test': np.array(results['y_test']),
        'preds': np.array(results['preds']),
        'f1_macro': results.get('f1_macro'),
        'f1_binary': results.get('f1_binary'),
        'num_steps': data.get('train_config', {}).get('num_steps', 0),
    }


def main(args):
    results_base = args.results_dir
    task_name = "cat_bench_regression"
    sample_dir = f"samples={args.sample_type}"

    # Find all baseline results
    baseline_dir = os.path.join(results_base, task_name, sample_dir, "baseline")
    if not os.path.exists(baseline_dir):
        print(f"No baseline directory found at {baseline_dir}")
        return

    baselines = {}
    for model_name in os.listdir(baseline_dir):
        baseline_results_path = os.path.join(baseline_dir, model_name, "0", "results.json")
        if os.path.exists(baseline_results_path):
            data = load_results(baseline_results_path)
            if data is not None:
                baselines[model_name] = data
                print(f"Loaded baseline: {model_name} (F1 macro={data['f1_macro']:.4f})")
            else:
                print(f"Warning: baseline {model_name} missing y_test/preds, skipping")

    if not baselines:
        print("No valid baselines found (need y_test and preds in results.json). Re-run evaluation first.")
        return

    # Find all trained model results
    trained_dir = os.path.join(results_base, task_name, sample_dir)
    all_results_files = glob(os.path.join(trained_dir, "**", "results.json"), recursive=True)

    # Group by model run (exclude baselines)
    model_runs = {}
    for rf in all_results_files:
        if "/baseline/" in rf:
            continue
        data = load_results(rf)
        if data is None:
            continue
        # Get the relative path from trained_dir to identify the model run
        rel = os.path.relpath(os.path.dirname(rf), trained_dir)
        # The model run key is everything except the last checkpoint number
        parts = rel.split(os.sep)
        # Last part is the checkpoint step number
        run_key = os.sep.join(parts[:-1])
        if run_key not in model_runs:
            model_runs[run_key] = []
        model_runs[run_key].append({
            'path': rf,
            'rel': rel,
            'data': data,
        })

    # For each model run, find matching baseline and run bootstrap
    all_bootstrap_results = {}
    for run_key, checkpoints in sorted(model_runs.items()):
        # Determine base model name from the run key
        base_model = find_baseline_model_name(run_key)
        if base_model is None or base_model not in baselines:
            print(f"Warning: could not find baseline for {run_key} (detected: {base_model}), skipping")
            continue

        baseline_data = baselines[base_model]
        print(f"\n{'='*80}")
        print(f"Model run: {run_key}")
        print(f"Baseline: {base_model} (F1 macro={baseline_data['f1_macro']:.4f})")
        print(f"{'='*80}")

        checkpoints = sorted(checkpoints, key=lambda x: x['data']['num_steps'])
        run_results = []

        for ckpt in checkpoints:
            d = ckpt['data']
            bt = bootstrap_f1_delta(
                baseline_data['y_test'],  # y_true is the same for both
                d['preds'],
                baseline_data['preds'],
                metric=args.metric,
                B=args.n_bootstrap,
            )
            bt['num_steps'] = d['num_steps']
            bt['f1_model'] = d[args.metric]
            bt['f1_baseline'] = baseline_data[args.metric]
            run_results.append(bt)

            sig = "*" if bt['significant_at_0.05'] else " "
            print(f"  step={d['num_steps']:>7d}  F1={d[args.metric]:.4f}  "
                  f"delta={bt['mean_delta']:+.4f}  CI=[{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
                  f"p={bt['p_value']:.4f} {sig}")

        all_bootstrap_results[run_key] = {
            'baseline': base_model,
            'metric': args.metric,
            'n_bootstrap': args.n_bootstrap,
            'checkpoints': run_results,
        }

    # Save all bootstrap results
    out_path = os.path.join(results_base, task_name, sample_dir, "bootstrap_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_bootstrap_results, f, indent=2)
    print(f"\nBootstrap results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired bootstrap test for CaT-Bench regression results")
    parser.add_argument("--results_dir", default="./results_stat_test",
                        help="Base results directory containing cat_bench_regression results")
    parser.add_argument("--sample_type", default="all", choices=["real", "all"])
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "f1_binary"])
    parser.add_argument("--n_bootstrap", default=2000, type=int, help="Number of bootstrap iterations")
    args = parser.parse_args()
    main(args)
