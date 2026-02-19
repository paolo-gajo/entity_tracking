import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tensor_heatmap(tensor, filename = 'tensor.pdf', title="Tensor Heatmap", cmap="viridis", text_color="white", dpi = 100):
    """
    Plots an NxN PyTorch tensor as a heatmap.
    
    Args:
        tensor: The input tensor (e.g., torch.Size([6, 6]))
        title: Title of the plot
        cmap: Matplotlib colormap (e.g., 'viridis', 'coolwarm', 'magma')
        text_color: Color of the annotation text
    """
    
    # 1. Preprocessing: Move to CPU, detach gradients, convert to numpy
    if isinstance(tensor, torch.Tensor):
        # Handle cases where tensor is on GPU or has requires_grad=True
        data = tensor.detach().cpu().numpy()
    else:
        data = tensor # Fallback for numpy arrays

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Render the heatmap
    im = ax.imshow(data, cmap=cmap)

    # 3. Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    # 4. Add Ticks
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    
    # Label axes starting from 0
    ax.set_xticklabels(np.arange(data.shape[1]))
    ax.set_yticklabels(np.arange(data.shape[0]))

    # 5. Add Text Annotations (Loop through cells)
    # Only do this if the tensor is reasonably small to prevent clutter
    # if data.shape[0] <= 20 and data.shape[1] <= 20:
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             # Formatting float to 2 decimal places
    #             text = ax.text(j, i, f"{data[i, j]:.2f}",
    #                            ha="center", va="center", color=text_color, fontweight='bold')

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename, dpi = dpi)


def save_heatmaps(S_directed, S_undirected, suffix = ''):
    S_directed_save_path = os.path.join('./heatmaps', f'S_directed{suffix}.png')
    S_undirected_save_path = os.path.join('./heatmaps', f'S_undirected{suffix}.png')
    plot_tensor_heatmap(S_directed, S_directed_save_path, dpi = 100)
    plot_tensor_heatmap(S_undirected, S_undirected_save_path, dpi = 10)

def main():

    # Create a random 6x6 tensor
    N = 6
    my_tensor = torch.randn(N, N)

    # Call the function
    plot_tensor_heatmap(my_tensor, title=f"{N}x{N} Tensor Heatmap")

if __name__ == "__main__":
    main()