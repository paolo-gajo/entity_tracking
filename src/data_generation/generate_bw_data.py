"""
generate_bw_data.py — Standalone Blocks World dataset generator.

Generates BW problems, plans, DAGs, and outputs the dataset in the
{orig, shuf, binary_label} JSON format consumed by pretrain_bw.py.

Can also dump a few samples to stdout for inspection.

Usage:
    python src/generate_bw_data.py --n_problems 50000 --save_path ./data/bw/bw_stp_dataset.json
    python src/generate_bw_data.py --n_problems 5 --inspect  # print samples
"""

from __future__ import annotations
import random
import json
import os
import argparse


# ── Compact JSON encoder: keeps leaf-level lists on a single line ──

class CompactJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that prints leaf-level lists (lists whose elements are
    all primitives or all short sub-lists of primitives) on a single line
    while still indenting dicts and nested structures normally.

    Produces:
        "dag_edges": [[0, 1], [0, 10], [2, 5]],
        "goal_tower": ["B", "G", "C", "E", "A", "F", "D"],
    instead of one element per line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent_val = self.indent  # save original indent

    def _is_leaf_list(self, obj):
        """True if obj is a list that should be rendered in one line."""
        if not isinstance(obj, list):
            return False
        for item in obj:
            if isinstance(item, (dict,)):
                return False
            if isinstance(item, list):
                # allow one level of nesting (e.g. [[0,1],[2,3]])
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
            inner = ",\n".join(items)
            return "{\n" + inner + "\n" + indent_str + "}"

        if isinstance(obj, list):
            if not obj:
                return "[]"
            if self._is_leaf_list(obj):
                # Render entire list on one line
                parts = [json.dumps(item, ensure_ascii=self.ensure_ascii) for item in obj]
                return "[" + ", ".join(parts) + "]"
            # Non-leaf list: one element per line (but each element may itself be compact)
            items = []
            for item in obj:
                val_str = self._encode(item, level + 1)
                items.append(f"{child_indent}{val_str}")
            inner = ",\n".join(items)
            return "[\n" + inner + "\n" + indent_str + "]"

        return json.dumps(obj, ensure_ascii=self.ensure_ascii)

import networkx as nx

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ── Block names ──

def make_block_names(n: int) -> list[str]:
    if n <= 26:
        return [chr(ord("A") + i) for i in range(n)]
    return [f"B{i + 1}" for i in range(n)]


# ── Problem generation ──

def generate_bw_problem(n_blocks: int, n_towers: int):
    blocks = make_block_names(n_blocks)
    goal_tower = list(blocks)
    random.shuffle(goal_tower)

    shuffled = list(blocks)
    random.shuffle(shuffled)
    n_towers = min(n_towers, n_blocks)
    towers = [[] for _ in range(n_towers)]
    for i, b in enumerate(shuffled):
        towers[i % n_towers].append(b)
    towers = [t for t in towers if t]
    return towers, goal_tower


# ── Plan + DAG ──

def get_step_preconditions_and_effects(step):
    """
    Return (preconditions, add_effects, delete_effects) for one plan step.

    Each step in the data is really TWO basic blocksworld moves glued together:

      "Remove X from Y"  =  unstack(X, Y)  then  putdown(X)
      "Stack X onto Y"   =  pickup(X)      then  stack(X, Y)

    We figure out the NET result of the pair:
      preconditions  – what must be true BEFORE the step
      add_effects    – what is true AFTER the step
      delete_effects – what is NO LONGER true after the step

    We leave out "arm_empty" because every step both needs it and restores
    it, so it never forces one step to come before another.

    Each fact is a tuple like ('on', 'A', 'B') or ('clear', 'A') or
    ('on_table', 'A').
    """
    block = step["block"]

    if step["phase"] == "unstack":
        #   "Remove block X from block Y and place it on the table."
        #
        #   BEFORE the step we need:
        #     - X sits on Y          ('on', X, Y)
        #     - nothing is on top of X ('clear', X)
        #
        #   AFTER the step:
        #     - X is on the table     ('on_table', X)   ← NEW
        #     - Y is now clear        ('clear', Y)      ← NEW  (X was removed)
        #     - X is still clear      ('clear', X)      ← preserved
        #     - X is no longer on Y   ('on', X, Y)      ← GONE
        #
        Y = step["from_"]
        preconditions  = {("on", block, Y), ("clear", block)}
        add_effects    = {("on_table", block), ("clear", block), ("clear", Y)}
        delete_effects = {("on", block, Y)}

    else:  # "stack"
        #   "Stack block X onto block Y."
        #
        #   BEFORE the step we need:
        #     - X is on the table     ('on_table', X)
        #     - nothing is on top of X ('clear', X)
        #     - nothing is on top of Y ('clear', Y)
        #
        #   AFTER the step:
        #     - X sits on Y           ('on', X, Y)      ← NEW
        #     - X is still clear      ('clear', X)      ← preserved
        #     - X is no longer on table ('on_table', X)  ← GONE
        #     - Y is no longer clear   ('clear', Y)      ← GONE  (X is on Y)
        #
        Y = step["to"]
        preconditions  = {("on_table", block), ("clear", block), ("clear", Y)}
        add_effects    = {("on", block, Y), ("clear", block)}
        delete_effects = {("on_table", block), ("clear", Y)}

    return preconditions, add_effects, delete_effects


def compute_plan_and_dag(initial_towers, goal_tower):
    """
    Compute the plan (list of steps) and the dependency DAG.

    The DAG tells you which steps MUST happen before which other steps.
    If two steps are NOT connected by any path in the DAG, they are
    independent — you can do them in either order and the plan still works.

    HOW WE BUILD THE DAG — the EOG algorithm
    (Explanation-Based Order Generalization):

    The idea is simple.  For each step, we ask: "where does each of its
    preconditions come from?"  That gives us CAUSAL LINKS — edges that
    say "step j must happen before step i because j produces something
    i needs."

    Then we check for THREATS: if some other step k could destroy a fact
    that a causal link is carrying, we add an extra edge to make sure k
    doesn't sneak in between the step that provides it and the step
    that needs it.

    That's it — causal links + threat protection = correct DAG.
    """

    # ── Generate the plan steps (same logic as before) ──────────────

    support = {}
    for tower in initial_towers:
        for i, block in enumerate(tower):
            support[block] = "TABLE" if i == 0 else tower[i - 1]

    in_position = set()
    for i, block in enumerate(goal_tower):
        expected = "TABLE" if i == 0 else goal_tower[i - 1]
        if support.get(block) == expected and (i == 0 or goal_tower[i - 1] in in_position):
            in_position.add(block)
        else:
            break

    steps = []

    # Phase 1: unstack — take apart towers from top to bottom
    for tower in initial_towers:
        for block in reversed(tower):
            if block in in_position:
                break
            if support[block] == "TABLE":
                continue
            steps.append({
                "block": block, "from_": support[block], "to": "TABLE",
                "phase": "unstack",
                "text": f"Remove block {block} from block {support[block]} and place it on the table.",
            })

    # Phase 2: stack — build the goal tower from bottom to top
    start = len(in_position)
    for i in range(start, len(goal_tower)):
        block = goal_tower[i]
        target = goal_tower[i - 1] if i > 0 else "TABLE"
        if target == "TABLE":
            continue
        steps.append({
            "block": block, "from_": "TABLE", "to": target,
            "phase": "stack",
            "text": f"Stack block {block} onto block {target}.",
        })

    if len(steps) < 2:
        return steps, nx.DiGraph(), in_position

    # ── Build the dependency DAG using EOG ──────────────────────────

    # STEP 1: What is true in the world BEFORE any step runs?
    #
    # We read this off the initial tower configuration.
    # Example: towers [["D","C","E"], ["B","A"]] gives us:
    #   ('on_table', 'D'), ('on', 'C', 'D'), ('on', 'E', 'C'), ('clear', 'E')
    #   ('on_table', 'B'), ('on', 'A', 'B'), ('clear', 'A')

    initial_state = set()
    for tower in initial_towers:
        initial_state.add(("on_table", tower[0]))        # bottom block on table
        for i in range(1, len(tower)):
            initial_state.add(("on", tower[i], tower[i - 1]))  # each block on the one below
        initial_state.add(("clear", tower[-1]))           # top block is clear

    # STEP 2: For each step, figure out what it needs and what it changes.

    n = len(steps)
    step_pre = []   # step_pre[i]  = set of precondition facts for step i
    step_add = []   # step_add[i]  = set of facts that become true after step i
    step_del = []   # step_del[i]  = set of facts that become false after step i

    for s in steps:
        pre, add, delete = get_step_preconditions_and_effects(s)
        step_pre.append(pre)
        step_add.append(add)
        step_del.append(delete)

    # STEP 3: Build the DAG — one node per step, edges = ordering constraints.

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))

    for i in range(n):
        for p in step_pre[i]:

            # ── 3a. Find the SUPPORTER of precondition p for step i ─────
            #
            # The supporter is the latest step BEFORE i whose add effects
            # include p.  It's the step that "provides" p to step i.
            #
            # If no earlier step provides p, then p must already be true
            # in the initial state (before any steps run).

            supporter = None                        # None means "initial state"
            for j in range(i - 1, -1, -1):          # scan backwards from i
                if p in step_add[j]:
                    supporter = j
                    break

            # Add a causal-link edge:  supporter ──→ i
            # meaning "supporter must happen before i"
            if supporter is not None:
                dag.add_edge(supporter, i)

            # ── 3b. Protect this causal link from THREATS ───────────────
            #
            # A threat is any step k that DELETES p.  If k were to run
            # between the supporter and step i, it would destroy p and
            # step i would fail.
            #
            # To prevent that, we force k to be OUTSIDE the window:
            #   - either k runs before the supporter (so the supporter
            #     re-establishes p after k destroys it), or
            #   - k runs after step i (so step i uses p before k
            #     destroys it).
            #
            # Example from the bug we found:
            #   step 3 = "Remove G from A"  needs ('clear', 'G')
            #   step 4 = "Stack D onto G"   deletes ('clear', 'G')
            #   Without threat protection, the DAG would allow step 4
            #   before step 3, and G would have D on top → can't remove G.
            #   Threat protection adds edge 3→4: step 3 before step 4.

            for k in range(n):
                if k == i or k == supporter:
                    continue
                if p not in step_del[k]:
                    continue

                # Step k deletes p — it's a threat.  Force it outside.

                if supporter is None:
                    # p comes from the initial state (before all steps).
                    # We can't put k "before the initial state", so the
                    # only option is: k must come AFTER step i.
                    dag.add_edge(i, k)

                elif k < supporter:
                    # k is before the supporter in the original plan.
                    # Keep it that way: k must stay before the supporter.
                    dag.add_edge(k, supporter)

                elif k > i:
                    # k is after step i in the original plan.
                    # Keep it that way: k must stay after step i.
                    dag.add_edge(i, k)

                # The remaining case (supporter < k < i) is impossible
                # in a correct plan — it would mean k destroys p between
                # the step that provides it and the step that needs it,
                # so the plan wouldn't work.

    # STEP 4: Remove redundant edges.
    #
    # If we have A→B and B→C, we don't also need A→C — it's implied.
    # The "transitive reduction" strips out these redundant edges,
    # leaving only the essential ones.

    dag = nx.transitive_reduction(dag)

    assert nx.is_directed_acyclic_graph(dag), "BUG: produced a cyclic graph!"
    return steps, dag, in_position


# ── Toposort sampling ──

def sample_toposort(dag):
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


def is_valid_toposort(ordering, dag):
    pos = {node: i for i, node in enumerate(ordering)}
    return all(pos[u] < pos[v] for u, v in dag.edges())


def sample_invalid_ordering(dag, max_attempts=200):
    nodes = list(dag.nodes())
    for _ in range(max_attempts):
        perm = list(nodes)
        random.shuffle(perm)
        if not is_valid_toposort(perm, dag):
            return perm
    # Fallback: swap a dependent pair
    valid = sample_toposort(dag)
    edges = list(dag.edges())
    if not edges:
        return None
    random.shuffle(edges)
    u, v = edges[0]
    perm = list(valid)
    iu, iv = perm.index(u), perm.index(v)
    perm[iu], perm[iv] = perm[iv], perm[iu]
    return perm if not is_valid_toposort(perm, dag) else None


def count_valid_toposorts(dag, limit=10000):
    count = 0
    for _ in nx.all_topological_sorts(dag):
        count += 1
        if count >= limit:
            break
    return count


def get_violated_edges(ordering, dag):
    pos = {node: i for i, node in enumerate(ordering)}
    return [(u, v) for u, v in dag.edges() if pos[u] > pos[v]]


# ── PDDL ──

def to_pddl(initial_towers, goal_tower, problem_name="bw-problem"):
    all_blocks = sorted({b for t in initial_towers for b in t})
    init_facts = ["(armempty)"]
    for tower in initial_towers:
        for i, block in enumerate(tower):
            bl = block.lower()
            if i == 0:
                init_facts.append(f"(ontable {bl})")
            else:
                init_facts.append(f"(on {bl} {tower[i-1].lower()})")
        init_facts.append(f"(clear {tower[-1].lower()})")
    goal_facts = []
    for i, block in enumerate(goal_tower):
        bl = block.lower()
        if i == 0:
            goal_facts.append(f"(ontable {bl})")
        else:
            goal_facts.append(f"(on {bl} {goal_tower[i-1].lower()})")
    objs = " ".join(b.lower() for b in all_blocks)
    nl = "\n    "
    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain blocksworld)\n"
        f"  (:objects {objs})\n"
        f"  (:init\n    {nl.join(init_facts)})\n"
        f"  (:goal (and\n    {nl.join(goal_facts)})))\n"
    )


# ── Dataset builder ──

def generate_dataset(
    n_problems, n_blocks_min=6, n_blocks_max=15,
    n_towers_min=2, n_towers_max=5, neg_ratio=0.5,
    save_path=None,
):
    """
    Generate BW dataset.  Each entry stores the canonical step texts and DAG
    edges so that valid/invalid orderings can be resampled at training time.

    Entry format:
        steps       : list[str]   — step texts in canonical (index) order
        dag_edges   : list[[u,v]] — causal dependency edges
        binary_label: int         — 1 = positive (model sees valid→valid),
                                    0 = negative (model sees invalid→valid)

    The neg_ratio controls the proportion of negatives in the dataset.
    At training time the Dataset class resamples both the valid toposort
    (completion) and the invalid ordering (prefix) on every __getitem__.
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

        meta.append({
            "prob_idx": prob_idx,
            "n_blocks": n_blocks,
            "n_steps": len(steps),
            "dag_edges": edges,
            "dag_width": max(len(ac) for ac in nx.antichains(dag)) if dag.number_of_nodes() > 0 else 0,
            # "n_valid_toposorts": count_valid_toposorts(dag, limit=1000),
            "initial_towers": initial,
            "goal_tower": goal,
            "steps": step_texts,
            "binary_label": binary_label,
            "pddl": to_pddl(initial, goal, f"bw-{prob_idx}"),
        })

    print(f"Generated {len(data)} samples ({skipped} skipped).")

    if save_path:
        encoder = CompactJSONEncoder(indent=2, ensure_ascii=False)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf8") as f:
            f.write(encoder.encode(data))
            f.write("\n")
        meta_path = save_path.replace(".json", "_meta.json")
        with open(meta_path, "w", encoding="utf8") as f:
            f.write(encoder.encode(meta))
            f.write("\n")
        print(f"Saved to {save_path} and {meta_path}")

    return data, meta


# ── Pretty-print for inspection ──

def inspect_sample(entry, meta_entry):
    """Pretty-print a sample, demonstrating runtime resampling."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(entry["steps"])))
    dag.add_edges_from(entry["dag_edges"])

    print("=" * 80)
    print(f"Problem {meta_entry['prob_idx']}: "
          f"{meta_entry['n_blocks']} blocks, {meta_entry['n_steps']} steps, "
          f"DAG width={meta_entry['dag_width']}, "
          f"valid toposorts={meta_entry['n_valid_toposorts']}, "
          f"label={'POSITIVE' if entry['binary_label'] else 'NEGATIVE'}")
    print()

    print("Initial towers:", meta_entry["initial_towers"])
    print("Goal tower:    ", meta_entry["goal_tower"])
    print()

    print("Plan steps (canonical order):")
    for i, s in enumerate(entry["steps"]):
        print(f"  s{i}: {s}")
    print()

    print(f"DAG edges: {entry['dag_edges']}")
    print()

    # Demonstrate resampling: show 3 different valid toposorts
    print("Example valid toposorts (resampled each time):")
    for k in range(3):
        ts = sample_toposort(dag)
        print(f"  toposort {k}: {ts}")
    print()

    # Demonstrate invalid ordering
    inv = sample_invalid_ordering(dag)
    if inv is not None:
        violated = get_violated_edges(inv, dag)
        print(f"Example invalid ordering: {inv}")
        print(f"  Violated edges: {[[u,v] for u,v in violated]}")
    print()

    print("PDDL:")
    print(meta_entry["pddl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_problems", type=int, default=10_000)
    parser.add_argument("--n_blocks_min", type=int, default=6)
    parser.add_argument("--n_blocks_max", type=int, default=15)
    parser.add_argument("--n_towers_min", type=int, default=2)
    parser.add_argument("--n_towers_max", type=int, default=5)
    parser.add_argument("--neg_ratio", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="./data/bw/bw_stp_dataset.json")
    parser.add_argument("--inspect", action="store_true",
                        help="Print a few samples to stdout instead of generating full dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.inspect:
        args.n_problems = 5
        args.save_path = None

    data, meta = generate_dataset(
        n_problems=args.n_problems,
        n_blocks_min=args.n_blocks_min,
        n_blocks_max=args.n_blocks_max,
        n_towers_min=args.n_towers_min,
        n_towers_max=args.n_towers_max,
        neg_ratio=args.neg_ratio,
        save_path=args.save_path,
    )

    if args.inspect:
        for entry, meta_entry in zip(data, meta):
            inspect_sample(entry, meta_entry)


if __name__ == "__main__":
    main()