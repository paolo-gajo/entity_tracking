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
