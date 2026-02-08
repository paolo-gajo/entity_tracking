import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import islice
import torch

def apply_step_order(t, step_order, step_indices):
    valid_positions = torch.where(step_indices != 0)
    t_shuffled = torch.zeros_like(t)
    buffer_tokens = torch.zeros_like(step_indices[valid_positions])
    buffer_indices = torch.zeros_like(step_indices[valid_positions])
    step_indices_shuffled = torch.zeros_like(step_indices)
    i = 0
    for j in step_order:
        step_positions = torch.where(step_indices == j)[0]
        selected_values = t[step_positions]
        shift = len(selected_values)
        buffer_tokens[i:i+shift] = selected_values
        buffer_indices[i:i+shift] = j
        i += shift
    t_shuffled[valid_positions] = buffer_tokens
    step_indices_shuffled[valid_positions] = buffer_indices
    return t_shuffled, step_indices_shuffled

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
