"""
sims_thinking.py — unshuffle probe.

Feed a thinking-capable LM the steps of a recipe in a scrambled order inside a
chat prompt, ask it to reason in <think></think> and then emit the steps in the
correct order as:

    Step 1: <verbatim step text>
    Step 2: <verbatim step text>
    ...

One forward pass over (prompt + completion) gives hidden states from which we
extract mean-pooled step embeddings from two sides:

  - prompt-side: the scrambled block in the user message, markers (A) (B) ...
  - completion-side: the model's reordering, markers "Step 1: " "Step 2: " ...

For each recipe we compute:
  - exact-position accuracy and Kendall tau between the model's predicted order
    and the original (as-written) gold order
  - ROC-AUC (directed and undirected, raw S and widest-path R) of the step-pair
    scores against the gold reachability matrix A, for both sides

Completion-side embeddings are re-indexed back to gold identity via exact-string
match of each emitted step against the gold step texts. Recipes where matching
fails to produce a bijection are dropped.

Finally we report per-AUC Pearson/Spearman correlations with exact-accuracy and
with Kendall tau across recipes.
"""

import argparse
import json
import os
import bisect
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Remove whitespace before punctuation so that the model's naturally
# detokenized output can match the erfgc space-joined word form.
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?)\]\}])")


def normalize_text(s):
    return _PUNCT_SPACE_RE.sub(r"\1", s.strip())

import networkx as nx
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Data prep
# -------------------------


def load_raw(json_paths):
    data = []
    for p in json_paths:
        with open(p, "r", encoding="utf8") as f:
            data += json.load(f)
    return data


def build_step_dag(head_indices, step_indices):
    """Step-level DAG using utils_data.make_edge_list's convention."""
    G = nx.DiGraph()
    n_steps = max(step_indices)
    G.add_nodes_from(range(1, n_steps + 1))
    hi = [0] + list(head_indices)
    si = [0] + list(step_indices)
    for i, j in enumerate(hi):
        src = si[i]
        tgt = si[j]
        if src != tgt and tgt != 0:
            G.add_edge(src, tgt)
    return G


def build_recipes(raw, min_longest_path=2):
    recipes = []
    for r in raw:
        si = r["step_indices"]
        n_steps = max(si)
        words = r["words"]
        steps = [
            " ".join(words[i] for i in range(len(words)) if si[i] == s)
            for s in range(1, n_steps + 1)
        ]
        G = build_step_dag(r["head_indices"], si)
        if not nx.is_directed_acyclic_graph(G):
            continue
        if nx.dag_longest_path_length(G) < min_longest_path:
            continue
        recipes.append({"steps": steps, "G": G})
    return recipes


# -------------------------
# Prompt construction
# -------------------------


SCRAMBLED_ANCHOR = "Scrambled steps:\n"


def build_unshuffle_prompt(tokenizer, steps, shuffled_order, thinking=True):
    """shuffled_order is a list of gold step indices (1-based) in the order
    they should appear in the prompt's scrambled block."""
    scrambled_block = "\n".join(
        f"Step {k + 1}: {steps[gold_idx - 1]}"
        for k, gold_idx in enumerate(shuffled_order)
    )

    user = (
        "You are given the steps of a cooking recipe in a scrambled order. "
        "Reason about the dependencies between steps, then output the steps "
        "in the correct execution order, copying each step's text exactly as "
        "written below (preserving spacing and punctuation).\n\n"
        "Use this exact output format, one line per step:\n"
        "Step 1: <verbatim step text>\n"
        "Step 2: <verbatim step text>\n"
        "...\n"
        f"Step {len(steps)}: <verbatim step text>\n\n"
        f"{SCRAMBLED_ANCHOR}{scrambled_block}"
    )
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def random_shuffled_order(n, rng):
    order = list(range(1, n + 1))
    while True:
        perm = order[:]
        rng.shuffle(perm)
        if perm != order:
            return perm


# -------------------------
# Token/char span utilities
# -------------------------


def build_token_offsets(tokenizer, ids):
    """Return (full_text, offsets) where offsets has length len(ids)+1 and
    offsets[k] is the character index in full_text at which token k starts
    (offsets[-1] == len(full_text)).

    Built via cumulative decode so that spacing in full_text matches what the
    tokenizer would flat-decode — avoids the per-token-decode spacing drift
    that breaks BPE/SentencePiece models.
    """
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    offsets = [0]
    for k in range(1, len(ids_list) + 1):
        offsets.append(
            len(tokenizer.decode(ids_list[:k], skip_special_tokens=False))
        )
    full = tokenizer.decode(ids_list, skip_special_tokens=False)
    return full, offsets


def _char_to_tok_idx(offsets, c):
    """Return the token index k such that offsets[k] <= c < offsets[k+1]."""
    idx = bisect.bisect_right(offsets, c) - 1
    return max(0, min(idx, len(offsets) - 2))


def extract_spans_by_markers(full_text, offsets, markers_in_order, char_start=0):
    """Find each marker in full_text (in the given order, starting at
    char_start) and return a list of
    (marker_idx, tok_start, tok_end, char_after_marker, char_next_marker).

    Returns None if any marker is not found.
    """
    anchors = []
    search = char_start
    for mi, m in enumerate(markers_in_order):
        pos = full_text.find(m, search)
        if pos == -1:
            return None
        anchors.append((pos, pos + len(m), mi))
        search = pos + len(m)

    n_tokens = len(offsets) - 1
    out = []
    for k, (c_start, c_end, mi) in enumerate(anchors):
        tok_start = _char_to_tok_idx(offsets, c_end)
        # If c_end falls strictly inside a token, advance past it so we don't
        # include marker chars in the span.
        if tok_start < n_tokens and offsets[tok_start] < c_end:
            tok_start += 1
        if k + 1 < len(anchors):
            next_c = anchors[k + 1][0]
            tok_end = _char_to_tok_idx(offsets, next_c)
        else:
            tok_end = n_tokens
        char_next = anchors[k + 1][0] if k + 1 < len(anchors) else len(full_text)
        out.append((mi, tok_start, tok_end, c_end, char_next))
    return out


def mean_pool_span(hidden, tok_start, tok_end):
    tok_end = min(tok_end, hidden.shape[0])
    if tok_start >= tok_end:
        return None
    return hidden[tok_start:tok_end].mean(dim=0)


# -------------------------
# Scoring (same kernel as sims.py)
# -------------------------


def compute_scores_from_H(H):
    diff = H.unsqueeze(0) - H.unsqueeze(1)
    penalty = torch.relu(diff).pow(2).sum(dim=-1)
    S_directed = -penalty
    Hc = H - H.mean(dim=0, keepdim=True)
    Hn = Hc / (Hc.norm(dim=1, keepdim=True) + 1e-8)
    S_undirected = Hn @ Hn.T
    return S_directed, S_undirected


def widest_path_closure(S):
    R = S.clone()
    n = R.shape[0]
    R.fill_diagonal_(-1e9)
    for k in range(n):
        via = torch.minimum(R[:, k].unsqueeze(1), R[k, :].unsqueeze(0))
        R = torch.maximum(R, via)
    R.fill_diagonal_(-1e9)
    return R


def gold_reachability_matrix(G, step_order):
    G_tc = nx.transitive_closure(G)
    A = nx.to_numpy_array(G_tc, nodelist=step_order).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A


def auc_against_A(S, A):
    S_np = S.detach().cpu().to(torch.float32).numpy()
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = A.astype(int)[mask]
    y_score = S_np[mask]
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def all_aucs(H, G, use_gold_transpose=False):
    S_dir, S_undir = compute_scores_from_H(H)
    R_dir = widest_path_closure(S_dir)
    R_undir = widest_path_closure(S_undir)
    n_steps = H.shape[0]
    A_gold = gold_reachability_matrix(G, list(range(1, n_steps + 1)))
    A_eval = A_gold.T if use_gold_transpose else A_gold
    return {
        "directed": auc_against_A(R_dir, A_eval),
        "undirected": auc_against_A(R_undir, A_eval),
        "directed_raw": auc_against_A(S_dir, A_eval),
        "undirected_raw": auc_against_A(S_undir, A_eval),
    }


# -------------------------
# Order metrics
# -------------------------


def exact_position_accuracy(predicted_order):
    """predicted_order: list where position k holds the gold step index the
    model put at position k (1-based). Gold order is 1..n."""
    n = len(predicted_order)
    return sum(1 for k, g in enumerate(predicted_order) if g == k + 1) / n


def kendall_tau_against_identity(predicted_order):
    n = len(predicted_order)
    if n < 2:
        return float("nan")
    gold = list(range(1, n + 1))
    tau, _ = stats.kendalltau(gold, predicted_order)
    return float(tau) if tau is not None and not np.isnan(tau) else float("nan")


# -------------------------
# Core per-recipe processing
# -------------------------


@dataclass
class RecipeResult:
    n_steps: int
    alignment_ok: bool
    exact_acc: float = float("nan")
    kendall_tau: float = float("nan")
    prompt_aucs: dict = field(default_factory=dict)
    completion_aucs: dict = field(default_factory=dict)
    predicted_order: Optional[List[int]] = None
    generated_text: str = ""


def process_recipe(
    recipe,
    tokenizer,
    model,
    device,
    rng,
    thinking=True,
    max_new_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    do_sample=True,
):
    steps = recipe["steps"]
    G = recipe["G"]
    n = len(steps)
    hidden_dim = model.config.hidden_size

    shuffled_order = random_shuffled_order(n, rng)  # gold idxs in prompt order
    prompt_text = build_unshuffle_prompt(tokenizer, steps, shuffled_order, thinking=thinking)

    # --- 1. Generate ---
    tokenizer.padding_side = "left"
    gen_inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=4096,
    ).to(device)
    with torch.no_grad():
        gen_sequences = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    prompt_len = gen_inputs["input_ids"].shape[1]
    tokenizer.padding_side = "right"

    gen_ids = gen_sequences[0, prompt_len:]
    if tokenizer.eos_token_id is not None:
        eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            gen_ids = gen_ids[: eos_pos[0]]
    prompt_ids = gen_inputs["input_ids"][0]

    full_ids = torch.cat([prompt_ids, gen_ids], dim=0)
    model_max = getattr(model.config, "max_position_embeddings", 8192)
    full_ids = full_ids[:model_max]

    # --- 2. Forward pass over full sequence for hidden states ---
    with torch.no_grad():
        out = model(input_ids=full_ids.unsqueeze(0), output_hidden_states=True)
    hidden_full = out.hidden_states[-1][0]  # (seq_len, hidden_dim)

    prompt_hidden = hidden_full[: len(prompt_ids)]
    gen_hidden = hidden_full[len(prompt_ids) : len(prompt_ids) + len(gen_ids)]

    prompt_full, prompt_offsets = build_token_offsets(tokenizer, prompt_ids)
    gen_full, gen_offsets = build_token_offsets(tokenizer, gen_ids)

    # --- 3. Prompt-side step embeddings ---
    # Skip past the instructions' literal "Step 1: ..." format examples by
    # anchoring at the unique "Scrambled steps:\n" header.
    anchor_pos = prompt_full.find(SCRAMBLED_ANCHOR)
    prompt_char_start = (anchor_pos + len(SCRAMBLED_ANCHOR)) if anchor_pos != -1 else 0

    step_markers = [f"Step {k + 1}: " for k in range(n)]
    prompt_spans = extract_spans_by_markers(
        prompt_full, prompt_offsets, step_markers, char_start=prompt_char_start,
    )

    H_prompt = torch.zeros(n, hidden_dim, device=device, dtype=hidden_full.dtype)
    prompt_ok = prompt_spans is not None
    if prompt_ok:
        for (mi, tok_start, tok_end, _, _) in prompt_spans:
            gold_idx = shuffled_order[mi]  # 1-based gold step at prompt position mi
            pooled = mean_pool_span(prompt_hidden, tok_start, tok_end)
            if pooled is None:
                prompt_ok = False
                break
            H_prompt[gold_idx - 1] = pooled

    # --- 4. Completion-side step embeddings + order decoding ---
    # Skip past any "Step k:" mentions the model made inside the thinking
    # trace by anchoring at the end of </think>.
    think_end = gen_full.find("</think>")
    gen_char_start = (think_end + len("</think>")) if think_end != -1 else 0

    comp_spans = extract_spans_by_markers(
        gen_full, gen_offsets, step_markers, char_start=gen_char_start,
    )

    alignment_ok = comp_spans is not None
    predicted_order = None
    H_completion = torch.zeros(n, hidden_dim, device=device, dtype=hidden_full.dtype)

    if alignment_ok:
        # Normalize gold to match the model's natural (detokenized) spacing.
        gold_norm = {normalize_text(steps[g - 1]): g for g in range(1, n + 1)}
        if len(gold_norm) != n:
            alignment_ok = False
        else:
            predicted_order = []
            used = set()
            for (mi, tok_start, tok_end, char_after, char_next) in comp_spans:
                slice_text = gen_full[char_after:char_next]
                key = normalize_text(slice_text)
                if key not in gold_norm:
                    alignment_ok = False
                    break
                g = gold_norm[key]
                if g in used:
                    alignment_ok = False
                    break
                used.add(g)
                predicted_order.append(g)
                pooled = mean_pool_span(gen_hidden, tok_start, tok_end)
                if pooled is None:
                    alignment_ok = False
                    break
                H_completion[g - 1] = pooled
            if alignment_ok and len(predicted_order) != n:
                alignment_ok = False

    gen_text = gen_full

    result = RecipeResult(
        n_steps=n,
        alignment_ok=alignment_ok,
        predicted_order=predicted_order,
        generated_text=gen_text,
    )

    if alignment_ok:
        result.exact_acc = exact_position_accuracy(predicted_order)
        result.kendall_tau = kendall_tau_against_identity(predicted_order)
        result.completion_aucs = all_aucs(H_completion, G)

    if prompt_ok:
        result.prompt_aucs = all_aucs(H_prompt, G)

    return result


# -------------------------
# Aggregation + correlations
# -------------------------


def correlate(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3 or np.std(xs[mask]) == 0 or np.std(ys[mask]) == 0:
        return {"n": int(mask.sum()), "pearson": float("nan"), "spearman": float("nan")}
    pr, _ = stats.pearsonr(xs[mask], ys[mask])
    sr, _ = stats.spearmanr(xs[mask], ys[mask])
    return {"n": int(mask.sum()), "pearson": float(pr), "spearman": float(sr)}


def aggregate(results: List[RecipeResult]):
    aligned = [r for r in results if r.alignment_ok]
    n_total = len(results)
    n_aligned = len(aligned)

    modes = ["directed", "undirected", "directed_raw", "undirected_raw"]
    sides = [("completion", lambda r: r.completion_aucs), ("prompt", lambda r: r.prompt_aucs)]

    out = {
        "n_total": n_total,
        "n_aligned": n_aligned,
        "alignment_rate": (n_aligned / n_total) if n_total else 0.0,
        "per_side": {},
    }

    for side_name, getter in sides:
        side_out = {"mean_auc": {}, "correlations_exact_acc": {}, "correlations_kendall_tau": {}}
        # Use all recipes with valid aucs for this side (prompt-side may be valid
        # even when completion alignment failed). But correlations require order
        # metrics, which require alignment_ok.
        for m in modes:
            vals = [getter(r).get(m, float("nan")) for r in results if getter(r)]
            vals = [v for v in vals if not np.isnan(v)]
            side_out["mean_auc"][m] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "n": len(vals),
            }
            xs_acc = [r.exact_acc for r in aligned if getter(r)]
            xs_tau = [r.kendall_tau for r in aligned if getter(r)]
            ys = [getter(r).get(m, float("nan")) for r in aligned if getter(r)]
            side_out["correlations_exact_acc"][m] = correlate(xs_acc, ys)
            side_out["correlations_kendall_tau"][m] = correlate(xs_tau, ys)
        out["per_side"][side_name] = side_out

    return out


# -------------------------
# Main
# -------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_files = [f"./data/erfgc/bio/{split}.json" for split in ["train", "val", "test"]]
    raw = load_raw(json_files)
    recipes = build_recipes(raw)
    if args.limit > 0:
        recipes = recipes[: args.limit]
    print(f"Loaded {len(recipes)} recipes after filtering.")

    print(f"Loading model: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    all_run_results = []
    for run_idx in range(args.n_runs):
        rng = random.Random(args.seed + run_idx)
        run_results: List[RecipeResult] = []
        for recipe in tqdm(recipes, desc=f"run {run_idx + 1}/{args.n_runs}"):
            try:
                res = process_recipe(
                    recipe, tokenizer, model, device, rng,
                    thinking=bool(args.thinking),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=bool(args.do_sample),
                )
            except Exception as e:
                print(f"recipe failed: {e}")
                continue
            run_results.append(res)
        all_run_results.append(run_results)

    # Aggregate across runs: treat each (recipe, run) pair as an independent
    # observation for mean AUC and for the correlations.
    flat = [r for run in all_run_results for r in run]
    summary = aggregate(flat)

    print(json.dumps(summary, indent=2))

    if args.save_results:
        model_leaf = os.path.basename(os.path.normpath(args.model_dir))
        save_dir = os.path.join(
            args.results_base_dir, "sims_thinking",
            f"thinking={args.thinking}", model_leaf,
        )
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "eval_config": vars(args),
            "summary": summary,
            "per_recipe": [
                {
                    "n_steps": r.n_steps,
                    "alignment_ok": r.alignment_ok,
                    "exact_acc": r.exact_acc,
                    "kendall_tau": r.kendall_tau,
                    "predicted_order": r.predicted_order,
                    "prompt_aucs": r.prompt_aucs,
                    "completion_aucs": r.completion_aucs,
                    "generated_text": r.generated_text if args.save_generations else "",
                }
                for r in flat
            ],
        }
        path = os.path.join(save_dir, "results.json")
        with open(path, "w", encoding="utf8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--thinking", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--do_sample", default=1, type=int)
    parser.add_argument("--limit", default=0, type=int,
                        help="If > 0, only evaluate first N recipes (debug)")
    parser.add_argument("--save_results", default=1, type=int)
    parser.add_argument("--save_generations", default=1, type=int)
    parser.add_argument("--results_base_dir", default="./results", type=str)
    args = parser.parse_args()
    main(args)
