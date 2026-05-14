import json
import argparse

def fmt_params(n):
    """Format parameter count as e.g. 124M."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)

def make_table(data):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lrl}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Model} & \textbf{Params} & \textbf{HuggingFace} \\")
    lines.append(r"    \midrule")
    for entry in data:
        model_id = entry["model_id"].replace("_", r"\_")
        params = fmt_params(entry["num_params"])
        url = entry["hf_url"]
        hf_link = rf"\href{{{url}}}{{\texttt{{{model_id}}}}}"
        lines.append(f"    {model_id} & {params} & {hf_link} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Pre-trained GPT-2 models used in experiments.}")
    lines.append(r"  \label{tab:models}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="model_params.json")
    parser.add_argument("--output", default=None, help="Write LaTeX to file instead of stdout")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    table = make_table(data)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved to {args.output}")
    else:
        print(table)
