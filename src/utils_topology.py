import networkx as nx
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import json
from itertools import permutations
from collections import defaultdict
import numpy as np
from itertools import islice

def batched_gen(iterable, n):
    iterator = iter(iterable)
    while True:
        batch = tuple(islice(iterator, n))
        if not batch:
            break
        yield batch

def save_graph_plot(G, save_dir, filename):
    pos = {node: (node, node % 2) for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_size=500, arrowstyle='-|>', arrowsize=20)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), format="pdf")
    plt.close()

def get_ordered_text(order_batch, words, step_indices):
    words = words[1:]
    step_indices = step_indices[1:]
    ordered_words_string_batch = []
    for order in order_batch:
        ordered_words_list = []
        for i in order:
            mask = step_indices == i
            ordered_words_list.append(words[mask])
        ordered_words_list = np.concatenate(ordered_words_list)
        ordered_words_string_batch.append(' '.join(ordered_words_list))
    return ordered_words_string_batch

class Grapher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def make_edge_list(self, tokens, head_indices, step_indices, prepend_zeroes = False):
        if prepend_zeroes:
            head_indices = [0] + head_indices
            step_indices = [0] + step_indices
        edges = []
        for i, j in enumerate(head_indices):
            src = step_indices[i]
            tgt = step_indices[j]
            if src != tgt and tgt != 0:
                edges.append((src, tgt))
        return edges

    def graph_from_erfgc(self, tokens, head_indices, step_indices):
        G = nx.DiGraph()
        edges = self.make_edge_list(tokens, head_indices=head_indices, step_indices=step_indices)
        G.add_edges_from(edges)
        # if len(G.nodes()) < max(step_indices) and len(G.nodes()) > 0:
        #     import pdb; pdb.set_trace()
        unique_steps = set(step_indices) - {0}
        G.add_nodes_from(unique_steps)
        # print(G.edges)
        return G