"""
pretrain_bw.py — Blocks World Step Token Prediction pre-training.

Generates Blocks World planning problems with single-tower goals,
extracts causal dependency DAGs from US (Unstack-Stack) plans, and
trains a language model using Step Token Prediction on valid/invalid
orderings — mirroring the STP regime in train.py (use_stp=1).

The key difference from recipe pre-training:
  - pi_orig is any valid topological ordering of the plan DAG
  - pi_shuf is an INVALID ordering (violates ≥1 DAG edge)
  - Positive samples use pi_shuf == pi_orig (a valid toposort)

Usage:
    python src/pretrain_bw.py \
        --model_name openai-community/gpt2 \
        --n_problems 50000 \
        --n_blocks_min 6 --n_blocks_max 15 \
        --batch_size 8 --lr 5e-5 --save_interval 1000

    # With CLM on text tokens too:
    python pretrain_bw.py --use_clm 1 --clm_lambda 1.0 ...
"""

from __future__ import annotations

import random
import json
import os
import argparse

import networkx as nx
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── Imports from existing codebase ────────────────────────────────────────
from utils_data import Collator
from utils_model import build_model_tokenizer
from utils_sys import save_run, setup_config
from loss_functions import (
    StepTokenLoss,
    CausalLMLoss,
    MaxMarginLoss,
    gather_losses,
)
from train.forward import compute_forward_bundle

torch.set_printoptions(linewidth=100000)


# ── Compact JSON encoder: keeps leaf-level lists on a single line ──

class CompactJSONEncoder(json.JSONEncoder):
    """Keeps leaf lists (primitives or short sub-lists) on one line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent_val = self.indent

    def _is_leaf_list(self, obj):
        if not isinstance(obj, list):
            return False
        for item in obj:
            if isinstance(item, dict):
                return False
            if isinstance(item, list):
                if not all(isinstance(x, (int, float, str, bool, type(None))) for x in item):
                    return False
            elif not isinstance(item, (int, float, str, bool, type(None))):
                return False
        return True

    def encode(self, obj):
        return self._encode(obj, level=0)

    def _encode(self, obj, level):
        indent_str = " " * (self.indent_val * level)
        child_indent = " " * (self.indent_val * (level + 1))
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for k, v in obj.items():
                key_str = json.dumps(k, ensure_ascii=self.ensure_ascii)
                val_str = self._encode(v, level + 1)
                items.append(f"{child_indent}{key_str}: {val_str}")
            return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"
        if isinstance(obj, list):
            if not obj:
                return "[]"
            if self._is_leaf_list(obj):
                parts = [json.dumps(item, ensure_ascii=self.ensure_ascii) for item in obj]
                return "[" + ", ".join(parts) + "]"
            items = [f"{child_indent}{self._encode(item, level + 1)}" for item in obj]
            return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"
        return json.dumps(obj, ensure_ascii=self.ensure_ascii)


# ══════════════════════════════════════════════════════════════════════════
#  1.  Blocks World Problem Generation
# ══════════════════════════════════════════════════════════════════════════

def make_block_names(n: int) -> list[str]:
    """Return n unique block names: A-Z for n ≤ 26, else B1..Bn."""
    if n <= 26:
        return [chr(ord("A") + i) for i in range(n)]
    return [f"B{i + 1}" for i in range(n)]


def generate_bw_problem(
    n_blocks: int,
    n_towers: int,
) -> tuple[list[list[str]], list[str]]:
    """
    Generate a random BW problem.

    Returns
    -------
    initial_towers : list[list[str]]
        Each inner list is a tower from bottom to top.
    goal_tower : list[str]
        Single goal tower from bottom to top.
    """
    blocks = make_block_names(n_blocks)

    # ── Goal: single tower in a random permutation ──
    goal_tower = list(blocks)
    random.shuffle(goal_tower)

    # ── Initial: distribute blocks randomly across n_towers towers ──
    shuffled = list(blocks)
    random.shuffle(shuffled)
    n_towers = min(n_towers, n_blocks)
    towers = [[] for _ in range(n_towers)]
    for i, b in enumerate(shuffled):
        towers[i % n_towers].append(b)

    # Drop any accidental empties (shouldn't happen, but safety)
    towers = [t for t in towers if t]
    return towers, goal_tower


# ──────────────────────────────────────────────────────────────────────────
#  Plan computation (US algorithm) + DAG extraction
# ──────────────────────────────────────────────────────────────────────────

def compute_plan_and_dag(
    initial_towers: list[list[str]],
    goal_tower: list[str],
) -> tuple[list[dict], nx.DiGraph, set[str]]:
    """
    Compute a US (Unstack-Stack) plan and its causal dependency DAG.

    The US plan:
      Phase 1 – unstack every misplaced block to the table (top-first per tower).
      Phase 2 – build the goal tower from bottom to top.

    Returns
    -------
    steps : list[dict]
        Each dict has keys: block, from_, to, phase, text.
    dag : nx.DiGraph
        Nodes are step indices 0..len(steps)-1.
        Edges represent causal dependencies (u → v means u before v).
    in_position : set[str]
        Blocks already in their goal position (no action needed).
    """
    # ── Support map: block → what it currently sits on ──
    support: dict[str, str] = {}
    for tower in initial_towers:
        for i, block in enumerate(tower):
            support[block] = "TABLE" if i == 0 else tower[i - 1]

    # ── In-position prefix of the goal tower ──
    in_position: set[str] = set()
    for i, block in enumerate(goal_tower):
        expected = "TABLE" if i == 0 else goal_tower[i - 1]
        if support.get(block) == expected and (i == 0 or goal_tower[i - 1] in in_position):
            in_position.add(block)
        else:
            break

    steps: list[dict] = []

    # ── Phase 1: unstack misplaced blocks to table (top-first) ──
    for tower in initial_towers:
        for block in reversed(tower):
            if block in in_position:
                break  # everything below is in position
            if support[block] == "TABLE":
                continue  # already on table, no move needed
            steps.append(
                {
                    "block": block,
                    "from_": support[block],
                    "to": "TABLE",
                    "phase": "unstack",
                    "text": f"Remove block {block} from block {support[block]} and place it on the table.",
                }
            )

    # ── Phase 2: build goal tower bottom-up ──
    start = len(in_position)
    for i in range(start, len(goal_tower)):
        block = goal_tower[i]
        target = goal_tower[i - 1] if i > 0 else "TABLE"
        # If this block's goal is "on the table", it's already there after
        # the unstack phase (US puts every misplaced block on the table).
        if target == "TABLE":
            continue
        steps.append(
            {
                "block": block,
                "from_": "TABLE",
                "to": target,
                "phase": "stack",
                "text": f"Stack block {block} onto block {target}.",
            }
        )

    if len(steps) < 2:
        return steps, nx.DiGraph(), in_position

    # ── Build dependency DAG ──
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(steps)))

    # Index maps
    unstack_step: dict[str, int] = {}  # block → step idx
    stack_step: dict[str, int] = {}
    for idx, s in enumerate(steps):
        if s["phase"] == "unstack":
            unstack_step[s["block"]] = idx
        else:
            stack_step[s["block"]] = idx

    # Rule 1: within-tower unstack ordering (top before bottom)
    for tower in initial_towers:
        chain = []
        for block in reversed(tower):
            if block in in_position:
                break
            if block in unstack_step:
                chain.append(unstack_step[block])
        for j in range(len(chain) - 1):
            dag.add_edge(chain[j], chain[j + 1])

    # Rule 2: stack chain (goal tower build order)
    stack_indices = [
        stack_step[goal_tower[i]]
        for i in range(len(in_position), len(goal_tower))
        if goal_tower[i] in stack_step
    ]
    for j in range(len(stack_indices) - 1):
        dag.add_edge(stack_indices[j], stack_indices[j + 1])

    # Rule 3: a block must be unstacked before it can be stacked
    for block, si in stack_step.items():
        if block in unstack_step:
            dag.add_edge(unstack_step[block], si)

    # Rule 4: if block B needs stacking and has a block X sitting on it
    #         in the initial state, X must be unstacked before B can be
    #         picked up to stack.
    on_top_of: dict[str, str] = {}
    for tower in initial_towers:
        for i in range(len(tower) - 1):
            on_top_of[tower[i]] = tower[i + 1]

    for block, si in stack_step.items():
        if block in on_top_of:
            above = on_top_of[block]
            if above in unstack_step:
                dag.add_edge(unstack_step[above], si)

    # Rule 5: if stacking X onto Y, and Y has a block above it in the
    #         initial state that needs unstacking, that unstack must
    #         precede stacking X.  (Handles the bottom-of-goal block
    #         that was skipped from stack_step.)
    for step_idx, s in enumerate(steps):
        if s["phase"] != "stack":
            continue
        target = s["to"]
        if target in on_top_of:
            above_target = on_top_of[target]
            if above_target in unstack_step:
                dag.add_edge(unstack_step[above_target], step_idx)

    assert nx.is_directed_acyclic_graph(dag), "BUG: dependency graph has a cycle"
    return steps, dag, in_position


# ──────────────────────────────────────────────────────────────────────────
#  Topological sort sampling
# ──────────────────────────────────────────────────────────────────────────

def sample_toposort(dag: nx.DiGraph) -> list[int]:
    """Random topological sort via Kahn's algorithm with random tie-breaking."""
    in_deg = dict(dag.in_degree())
    available = [n for n in dag.nodes() if in_deg[n] == 0]
    order = []
    while available:
        random.shuffle(available)
        chosen = available.pop()
        order.append(chosen)
        for succ in dag.successors(chosen):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                available.append(succ)
    return order


def is_valid_toposort(ordering: list[int], dag: nx.DiGraph) -> bool:
    """Check whether an ordering respects all DAG edges."""
    pos = {node: i for i, node in enumerate(ordering)}
    return all(pos[u] < pos[v] for u, v in dag.edges())


def sample_invalid_ordering(
    dag: nx.DiGraph,
    max_attempts: int = 200,
) -> list[int] | None:
    """
    Sample a random permutation of dag.nodes() that violates ≥1 edge.

    Strategy: random shuffle, reject if it happens to be valid.
    Falls back to swapping a dependent pair if pure shuffling keeps
    producing valid orderings (rare for wide DAGs).
    """
    nodes = list(dag.nodes())
    n = len(nodes)

    for _ in range(max_attempts):
        perm = list(nodes)
        random.shuffle(perm)
        if not is_valid_toposort(perm, dag):
            return perm

    # Fallback: take a valid toposort and swap one dependent pair
    valid = sample_toposort(dag)
    edges = list(dag.edges())
    if not edges:
        return None
    random.shuffle(edges)
    u, v = edges[0]
    perm = list(valid)
    iu, iv = perm.index(u), perm.index(v)
    perm[iu], perm[iv] = perm[iv], perm[iu]
    if not is_valid_toposort(perm, dag):
        return perm
    return None


def count_valid_toposorts(dag: nx.DiGraph, limit: int = 10000) -> int:
    """Count valid topological sorts (up to limit)."""
    count = 0
    for _ in nx.all_topological_sorts(dag):
        count += 1
        if count >= limit:
            break
    return count


# ──────────────────────────────────────────────────────────────────────────
#  PDDL generation (optional utility)
# ──────────────────────────────────────────────────────────────────────────

def to_pddl(
    initial_towers: list[list[str]],
    goal_tower: list[str],
    problem_name: str = "bw-problem",
) -> str:
    """Generate a PDDL problem string (standard 4-operator blocksworld)."""
    all_blocks = sorted({b for t in initial_towers for b in t})

    init_facts = ["(armempty)"]
    for tower in initial_towers:
        for i, block in enumerate(tower):
            bl = block.lower()
            if i == 0:
                init_facts.append(f"(ontable {bl})")
            else:
                init_facts.append(f"(on {bl} {tower[i - 1].lower()})")
        init_facts.append(f"(clear {tower[-1].lower()})")

    goal_facts = []
    for i, block in enumerate(goal_tower):
        bl = block.lower()
        if i == 0:
            goal_facts.append(f"(ontable {bl})")
        else:
            goal_facts.append(f"(on {bl} {goal_tower[i - 1].lower()})")

    objs = " ".join(b.lower() for b in all_blocks)
    nl = "\n    "
    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain blocksworld)\n"
        f"  (:objects {objs})\n"
        f"  (:init\n    {nl.join(init_facts)})\n"
        f"  (:goal (and\n    {nl.join(goal_facts)})))\n"
    )


# ══════════════════════════════════════════════════════════════════════════
#  2.  Dataset Generation
# ══════════════════════════════════════════════════════════════════════════

def generate_bw_dataset(
    n_problems: int,
    n_blocks_min: int = 6,
    n_blocks_max: int = 15,
    n_towers_min: int = 2,
    n_towers_max: int = 5,
    neg_ratio: float = 0.5,
    save_path: str | None = None,
) -> list[dict]:
    """
    Generate BW dataset in the {orig, shuf, binary_label} format expected
    by Seq2SeqDataset.make_step_token_pair_samples.

    Each entry:
      orig : list[str]   — step texts in a valid topological order
      shuf : list[str]   — step texts in an invalid order (neg) or same (pos)
      binary_label : int  — 1 if shuf is valid, 0 if invalid

    Also stores metadata (dag edges, pddl, etc.) for analysis/debugging.
    """
    data = []
    meta = []
    skipped = 0

    for prob_idx in tqdm(range(n_problems), desc="Generating BW problems"):
        n_blocks = random.randint(n_blocks_min, n_blocks_max)
        n_towers = random.randint(n_towers_min, min(n_towers_max, n_blocks))

        initial, goal = generate_bw_problem(n_blocks, n_towers)
        steps, dag, in_pos = compute_plan_and_dag(initial, goal)

        if len(steps) < 3:
            skipped += 1
            continue

        step_texts = [s["text"] for s in steps]
        edges = [[int(u), int(v)] for u, v in dag.edges()]

        is_positive = random.random() > neg_ratio
        binary_label = 1 if is_positive else 0

        data.append({
            "steps": step_texts,
            "dag_edges": edges,
            "binary_label": binary_label,
        })

        # Metadata (not used in training, useful for analysis)
        meta.append(
            {
                "prob_idx": prob_idx,
                "n_blocks": n_blocks,
                "n_steps": len(steps),
                "dag_edges": edges,
                "dag_width": max(
                    len(ac) for ac in nx.antichains(dag)
                ) if dag.number_of_nodes() > 0 else 0,
                "n_valid_toposorts": count_valid_toposorts(dag, limit=1000),
                "pddl": to_pddl(initial, goal, f"bw-{prob_idx}"),
                "initial_towers": initial,
                "goal_tower": goal,
                "binary_label": binary_label,
            }
        )

    print(f"Generated {len(data)} samples ({skipped} skipped).")

    if save_path:
        encoder = CompactJSONEncoder(indent=2, ensure_ascii=False)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf8") as f:
            f.write(encoder.encode(data))
            f.write("\n")
        print(f"Dataset saved to {save_path}")

        meta_path = save_path.replace(".json", "_meta.json")
        with open(meta_path, "w", encoding="utf8") as f:
            f.write(encoder.encode(meta))
            f.write("\n")
        print(f"Metadata saved to {meta_path}")

    return data


# ══════════════════════════════════════════════════════════════════════════
#  3.  BWDataset — resamples valid toposort on every __getitem__
# ══════════════════════════════════════════════════════════════════════════

from torch.utils.data import Dataset


class BWDataset(Dataset):
    """
    Dataset for Blocks World Step Token Prediction training.

    Each raw sample stores canonical step texts and DAG edges.  On every
    ``__getitem__`` call, a *fresh* valid topological sort is sampled as the
    completion target, and (for negatives) a fresh invalid ordering is
    sampled as the prefix.  This means the model is never trained toward a
    single arbitrary serialisation of the DAG.

    The per-sample formatting mirrors ``Seq2SeqDataset.make_step_token_pair_samples``
    from ``utils_data.py``, producing the same dict structure expected by the
    ``Collator.seq2seq_collate`` function.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        step_token_id_map: dict[int, int],
        max_length: int = 1024,
        loss_mask_type: str = "completion_only",
        prepend_bos: bool = False,
    ):
        self.tokenizer = tokenizer
        self.step_token_id_map = step_token_id_map
        self.max_length = max_length
        self.loss_mask_type = loss_mask_type
        self.prepend_bos = prepend_bos
        self.sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

        # Pre-tokenize step texts and rebuild DAGs once at init
        self.samples: list[dict] = []
        skipped = 0
        for item in data:
            n = len(item["steps"])
            if n < 2 or n > len(step_token_id_map):
                skipped += 1
                continue
            chunks = [
                tokenizer.encode(" " + s.strip(), add_special_tokens=False)
                for s in item["steps"]
            ]
            dag = nx.DiGraph()
            dag.add_nodes_from(range(n))
            dag.add_edges_from(item["dag_edges"])

            total_content_len = sum(len(c) for c in chunks) * 2 + n * 2 + len(self.sep_ids) + 2
            if total_content_len > max_length:
                skipped += 1
                continue

            self.samples.append({
                "chunks": chunks,
                "dag": dag,
                "n_steps": n,
                "binary_label": item["binary_label"],
            })
        if skipped:
            print(f"BWDataset: skipped {skipped} samples (too long or too many steps)")
        print(f"BWDataset: {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        chunks = item["chunks"]
        dag = item["dag"]
        n = item["n_steps"]
        binary_label = item["binary_label"]

        # ── Resample orderings ──
        valid_order = sample_toposort(dag)

        if binary_label == 1:
            shuf_order = list(valid_order)
        else:
            shuf_order = sample_invalid_ordering(dag)
            if shuf_order is None:
                # Extremely rare fallback: just reverse a valid sort
                shuf_order = list(reversed(valid_order))

        return self._format_sample(chunks, valid_order, shuf_order, n, binary_label)

    # ------------------------------------------------------------------

    def _format_sample(
        self,
        chunks: list[list[int]],
        valid_order: list[int],
        shuf_order: list[int],
        n: int,
        binary_label: int,
    ) -> dict:
        """
        Build the tokenised input dict for one sample.

        Mirrors ``Seq2SeqDataset.make_step_token_pair_samples`` but operates
        on index orderings instead of text lists, so toposorts can be
        resampled on every call.

        Layout:
            prefix  = [text_σ(0) <step_0>] [text_σ(1) <step_1>] … sep
            compl.  = [<step_j0> text_π(0)] [<step_j1> text_π(1)] … eos

        where σ = shuf_order, π = valid_order, and j_k is the prefix
        slot that holds step π(k).
        """
        stp = self.step_token_id_map

        # ── Prefix (shuffled order) ──────────────────────────────────
        prefix_ids: list[int] = []
        prefix_mml: list[int] = []
        loss_mask_prefix: list[int] = []

        for slot_j, canon_idx in enumerate(shuf_order):
            # Content tokens
            prefix_ids.extend(chunks[canon_idx])
            prefix_mml.extend([canon_idx + 1] * len(chunks[canon_idx]))
            loss_mask_prefix.extend([1] * len(chunks[canon_idx]))
            # Step token (slot_j determines which <step_j> is used)
            prefix_ids.append(stp[slot_j])
            prefix_mml.append(canon_idx + 1)
            loss_mask_prefix.append(0)

        # Separator
        prefix_ids.extend(self.sep_ids)
        prefix_mml.extend([0] * len(self.sep_ids))
        loss_mask_prefix.extend([1] * len(self.sep_ids))
        n_prefix = len(prefix_ids)

        # ── Completion (valid toposort order) ────────────────────────
        # Map: canonical step index → which prefix slot holds it
        canon_to_slot = {canon_idx: slot_j for slot_j, canon_idx in enumerate(shuf_order)}

        comp_ids: list[int] = []
        comp_mml: list[int] = []
        loss_mask_comp: list[int] = []
        stp_mask_comp: list[int] = []

        for canon_idx in valid_order:
            slot_j = canon_to_slot[canon_idx]
            # Step token (the model must predict which slot token comes next)
            comp_ids.append(stp[slot_j])
            comp_mml.append(canon_idx + 1)
            loss_mask_comp.append(0)
            stp_mask_comp.append(1)
            # Content tokens
            comp_ids.extend(chunks[canon_idx])
            comp_mml.extend([canon_idx + 1] * len(chunks[canon_idx]))
            loss_mask_comp.extend([1] * len(chunks[canon_idx]))
            stp_mask_comp.extend([0] * len(chunks[canon_idx]))

        # ── Assemble ─────────────────────────────────────────────────
        bos = [self.tokenizer.bos_token_id] if self.prepend_bos else []
        n_bos = len(bos)

        input_ids = bos + prefix_ids + comp_ids + [self.tokenizer.eos_token_id]
        attn_mask = [1] * len(input_ids)
        step_indices_mml = [0] * n_bos + prefix_mml + comp_mml + [0]

        if "completion_only" in self.loss_mask_type:
            loss_mask = [0] * (n_bos + n_prefix) + loss_mask_comp + [1]
            # Allow the last separator token to contribute to CLM so the
            # model learns to predict the first step token across the boundary.
            loss_mask[n_bos + n_prefix - 1] = 1
        elif "full" in self.loss_mask_type:
            loss_mask = [0] * n_bos + loss_mask_prefix + loss_mask_comp + [1]
        else:
            raise ValueError(f"Unknown loss_mask_type: {self.loss_mask_type}")

        stp_mask = [0] * (n_bos + n_prefix) + stp_mask_comp + [0]

        return {
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "loss_mask": loss_mask,
            "stp_mask": stp_mask,
            "step_indices_mml": step_indices_mml,
            "binary_label": binary_label,
        }


# ══════════════════════════════════════════════════════════════════════════
#  4.  Training Loop
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    # ── Resume handling ──
    resume_steps = 0
    if args.resume_from:
        ckpt_config_path = os.path.join(args.resume_from, "train_config.json")
        with open(ckpt_config_path, "r", encoding="utf8") as f:
            ckpt_config = json.load(f)
        resume_steps = ckpt_config["num_steps"]
        print(f"Resuming from checkpoint: {args.resume_from} (step {resume_steps})", flush=True)
        args.model_name = args.resume_from

    train_config = setup_config(args.__dict__)
    print(f"Train config:\n{json.dumps(train_config, indent=4)}", flush=True)

    # ── Generate or load BW data ──
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading existing dataset from {args.data_path}")
        with open(args.data_path, "r", encoding="utf8") as f:
            data_pairs = json.load(f)
    else:
        print("Generating Blocks World dataset...")
        save_path = args.data_path or "./data/bw/bw_stp_dataset.json"
        data_pairs = generate_bw_dataset(
            n_problems=args.n_problems,
            n_blocks_min=args.n_blocks_min,
            n_blocks_max=args.n_blocks_max,
            n_towers_min=args.n_towers_min,
            n_towers_max=args.n_towers_max,
            neg_ratio=args.neg_ratio,
            save_path=save_path,
        )
        args.data_path = save_path

    # Shuffle
    random.shuffle(data_pairs)

    if args.num_samples > 0:
        data_pairs = data_pairs[: args.num_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model + tokenizer + step tokens ──
    # build_model_tokenizer reads args.use_stp, args.stp_max_steps, etc.
    tokenizer, step_token_id_map, model, ref_model = build_model_tokenizer(args, device)

    # ── Dataset ──
    dataset = BWDataset(
        data_pairs,
        tokenizer,
        step_token_id_map=step_token_id_map,
        max_length=tokenizer.model_max_length,
        loss_mask_type=args.loss_mask_type,
        prepend_bos=bool(args.prepend_bos),
    )

    collator = Collator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator.seq2seq_collate,
        shuffle=True,
    )

    # ── Loss functions ──
    causal_lm_loss_fn = CausalLMLoss()
    stp_loss_fn = StepTokenLoss().to(device)
    kl_loss_fn = None

    hidden_dim = (
        model.config.hidden_size
        if hasattr(model.config, "hidden_size")
        else model.config.n_embd
    )
    max_margin_loss_fn = MaxMarginLoss(
        alpha=args.margin_alpha,
        activations=args.activations,
        hidden_dim=hidden_dim,
        proj_dim=args.mml_proj_dim,
    ).to(device)

    # ── Optimizer ──
    params = list(model.parameters())
    if max_margin_loss_fn.proj is not None:
        params += list(max_margin_loss_fn.proj.parameters())
    optimizer = AdamW(params=params, lr=args.lr)

    # ── Training loop ──
    tbar = tqdm(dataloader)
    num_steps = resume_steps
    losses = []
    prompt = None

    for batch_idx, batch in enumerate(tbar):
        if batch_idx < resume_steps:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        logits, lhs = compute_forward_bundle(args, model, batch)

        # STP loss (CE on step-token positions in the completion)
        stp_loss = stp_loss_fn(logits, batch["input_ids"], batch["stp_mask"])

        # Aggregate all losses through the standard gather_losses interface
        loss = gather_losses(
            args,
            causal_lm_loss_fn,
            kl_loss_fn,
            max_margin_loss_fn,
            logits,
            batch,
            device,
            lhs,
            pos_loss=None,
            stp_loss=stp_loss,
            cos_loss_fn=None,
        )

        optimizer.zero_grad()
        loss["total_loss"].backward()
        optimizer.step()

        stp_val = float(loss["stp_loss"].detach().cpu())
        clm_val = float(loss["causal_lm_loss"].detach().cpu())
        mml_val = float(loss["max_margin_loss"].detach().cpu())

        tbar.set_description(
            f"| CLM: {clm_val:.3f} "
            f"| MML: {mml_val:.3f} "
            f"| STP: {stp_val:.3f} "
        )

        losses.append(
            {
                "step": num_steps,
                "total": float(loss["total_loss"].detach().cpu()),
                "causal": clm_val,
                "mml": mml_val,
                "stp": stp_val,
            }
        )

        # Save a decoded prompt on the first step for debugging
        if num_steps == 0:
            from utils_data import prepare_text_batch_prompt

            prompt = prepare_text_batch_prompt(batch, tokenizer)
            os.makedirs("./misc", exist_ok=True)
            print(prompt, file=open("./misc/last_prompt_bw.txt", "w"), flush=True)

        num_steps += 1
        if num_steps % args.save_interval == 0:
            save_config = train_config.copy()
            save_config["num_steps"] = num_steps
            model_save_dir = os.path.join(train_config["model_save_dir"], str(num_steps))
            save_run(save_config, model_save_dir, model, tokenizer, prompt)

    # Save loss log
    json_path = os.path.join(train_config["model_save_dir"], "losses.json")
    if os.path.exists(train_config["model_save_dir"]):
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(losses, f, ensure_ascii=False, indent=4)


# ══════════════════════════════════════════════════════════════════════════
#  4.  CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train an LM on Blocks World using Step Token Prediction"
    )

    # ── BW generation ──
    parser.add_argument("--n_problems", type=int, default=50_000)
    parser.add_argument("--n_blocks_min", type=int, default=6)
    parser.add_argument("--n_blocks_max", type=int, default=15)
    parser.add_argument("--n_towers_min", type=int, default=2)
    parser.add_argument("--n_towers_max", type=int, default=5)
    parser.add_argument("--min_steps", type=int, default=3,
                        help="Minimum plan steps to keep a problem")

    # ── Model ──
    parser.add_argument("--model_name", default="openai-community/gpt2")
    parser.add_argument("--resume_from", default=None, type=str)

    # ── Data ──
    parser.add_argument("--data_path", default="./data/bw/bw_stp_dataset.json")
    parser.add_argument("--num_samples", default=0, type=int,
                        help="0 = use all generated samples")
    parser.add_argument("--neg_ratio", default=0.5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)

    # ── Prompt / mask ──
    parser.add_argument("--prompt_type", default="step_token_pairs")
    parser.add_argument("--attn_mask_type", default="full")
    parser.add_argument("--loss_mask_type", default="completion_only")
    parser.add_argument("--batch_mode", default="random_samples")
    parser.add_argument("--prepend_bos", default=0, type=int)

    # ── Losses ──
    # STP (always on for this script)
    parser.add_argument("--use_stp", default=1, type=int)
    parser.add_argument("--stp_lambda", default=1.0, type=float)
    parser.add_argument("--stp_max_steps", default=30, type=int,
                        help="M: number of step tokens (must be ≥ max plan length)")

    # CLM on text tokens (optional, can help ground the step tokens)
    parser.add_argument("--use_clm", default=0, type=int)
    parser.add_argument("--clm_lambda", default=1.0, type=float)
    parser.add_argument("--pool_clm", default=0, type=int)

    # MML (optional)
    parser.add_argument("--use_mml", default=0, type=int)
    parser.add_argument("--mml_lambda", default=0.1, type=float)
    parser.add_argument("--margin_alpha", default=0.05, type=float)
    parser.add_argument("--mml_proj_dim", default=0, type=int)
    parser.add_argument("--no_pos_mml", default=0, type=int)

    # KL / Cos / GRL (disabled by default; kept for interface compat)
    parser.add_argument("--use_kl", default=0, type=int)
    parser.add_argument("--kl_lambda", default=0.1, type=float)
    parser.add_argument("--use_cos", default=0, type=int)
    parser.add_argument("--cos_lambda", default=0.1, type=float)
    parser.add_argument("--cos_alpha", default=0.5, type=float)
    parser.add_argument("--use_grl", default=0, type=int)
    parser.add_argument("--pos_lambda", default=1.0, type=float)
    parser.add_argument("--grl_lambda", default=5.0, type=float)
    parser.add_argument("--pos_bins", default=32, type=int)
    parser.add_argument("--pos_head_hidden", default=256, type=int)
    parser.add_argument("--log_interval", default=100, type=int)

    # Misc
    parser.add_argument("--activations", default="real", type=str)
    parser.add_argument("--use_lora", default=0, type=int)
    parser.add_argument("--init_from_eos", default=0, type=int)
    parser.add_argument("--detect_anomaly", default=0, type=int)
    parser.add_argument("--save_heatmaps", default=0, type=int)

    args = parser.parse_args()

    # ── Validation ──
    assert args.use_stp == 1, "This script is designed for STP training (use_stp=1)"
    assert args.prompt_type == "step_token_pairs", (
        f"STP requires prompt_type='step_token_pairs', got '{args.prompt_type}'"
    )

    main(args)