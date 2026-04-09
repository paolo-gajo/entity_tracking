import argparse
import json
import os
import statistics


def _parse_seed_from_dirname(name):
    if not name.startswith("seed="):
        return None
    raw = name.split("=", 1)[1]
    try:
        return int(raw)
    except ValueError:
        return raw


def _seed_sort_key(seed_value):
    if isinstance(seed_value, int):
        return (0, seed_value)
    return (1, str(seed_value))


def _load_seed_results(results_root):
    per_seed = []
    missing = []

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    for name in os.listdir(results_root):
        seed = _parse_seed_from_dirname(name)
        if seed is None:
            continue

        seed_dir = os.path.join(results_root, name)
        if not os.path.isdir(seed_dir):
            continue

        result_file = os.path.join(seed_dir, "test_results.json")
        if not os.path.exists(result_file):
            missing.append({"seed": seed, "path": result_file})
            continue

        with open(result_file, "r", encoding="utf8") as f:
            data = json.load(f)

        per_seed.append(
            {
                "seed": seed,
                "path": result_file,
                "metrics": {
                    "step_accuracy": data.get("step_accuracy"),
                    "cat1_accuracy": data.get("cat1_accuracy"),
                    "cat2_f1": data.get("cat2_f1"),
                    "cat3_accuracy": data.get("cat3_accuracy"),
                    "cat1_total": data.get("cat1_total"),
                    "cat3_total": data.get("cat3_total"),
                },
            }
        )

    per_seed.sort(key=lambda x: _seed_sort_key(x["seed"]))
    missing.sort(key=lambda x: _seed_sort_key(x["seed"]))
    return per_seed, missing


def _compute_aggregate(per_seed):
    metric_names = [
        "step_accuracy",
        "cat1_accuracy",
        "cat2_f1",
        "cat3_accuracy",
        "cat1_total",
        "cat3_total",
    ]

    aggregate = {}
    for metric in metric_names:
        values = [entry["metrics"].get(metric) for entry in per_seed]
        values = [v for v in values if isinstance(v, (int, float))]

        if not values:
            aggregate[metric] = {"n": 0, "mean": None, "stdev": None}
            continue

        mean_val = float(statistics.mean(values))
        stdev_val = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        aggregate[metric] = {
            "n": len(values),
            "mean": mean_val,
            "stdev": stdev_val,
            "values": values,
        }

    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="./results/propara/openai-community_gpt2-medium",
        help="Root directory containing per-seed folders like seed=11",
    )
    parser.add_argument(
        "--output_name",
        default="seed_summary.json",
        help="Output JSON file name to save in results_root",
    )
    args = parser.parse_args()

    per_seed, missing = _load_seed_results(args.results_root)
    aggregate = _compute_aggregate(per_seed)

    summary = {
        "results_root": args.results_root,
        "num_seed_dirs_with_results": len(per_seed),
        "num_seed_dirs_missing_results": len(missing),
        "per_seed": per_seed,
        "missing_results": missing,
        "aggregate": aggregate,
    }

    output_path = os.path.join(args.results_root, args.output_name)
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {output_path}")
    for metric, stats in aggregate.items():
        if stats["n"] == 0:
            print(f"{metric}: no values")
        else:
            print(
                f"{metric}: mean={stats['mean']:.6f} stdev={stats['stdev']:.6f} n={stats['n']}"
            )


if __name__ == "__main__":
    main()
