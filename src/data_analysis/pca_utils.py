from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

# X is the n x k feature matrix your logistic regression sees
# y is the 0/1 labels

def run_and_plot_pca(filename_pca, G1, G2):
    if G1.dtype == torch.bfloat16:
        G1 = G1.to(dtype = torch.float16)
    if G2.dtype == torch.bfloat16:
        G2 = G2.to(dtype = torch.float16)
    if isinstance(G1, torch.Tensor): G1 = G1.cpu().numpy()
    if isinstance(G2, torch.Tensor): G2 = G2.cpu().numpy()

    X = np.vstack([G1, G2])
    print(X.shape)
    y = np.array([0]*len(G1) + [1]*len(G2))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')
    ax = fig.add_subplot()

    ax.scatter(X_pca[y==0, 0],
                X_pca[y==0, 1],
                # X_pca[y==0, 2],
                label='G1', alpha=.6,
                # c = X_pca[y==0, 3],
                # cmap = 'viridis',
                )
    ax.scatter(X_pca[y==1, 0],
                X_pca[y==1, 1],
                # X_pca[y==1, 2],
                label='G2', alpha=.6,
                # c = X_pca[y==1, 3],
                # cmap = 'viridis',
                )
    
    X_norm = (X - X.mean(axis = 0)) / X.std(axis = 0)
    
    pre_pca_centroid_g1_x = X_norm[y==0, 0].mean()
    pre_pca_centroid_g1_y = X_norm[y==0, 1].mean()
    pre_pca_centroid_g2_x = X_norm[y==1, 0].mean()
    pre_pca_centroid_g2_y = X_norm[y==1, 1].mean()

    post_pca_centroid_g1_x = X_pca[y==0, 0].mean()
    post_pca_centroid_g1_y = X_pca[y==0, 1].mean()
    post_pca_centroid_g2_x = X_pca[y==1, 0].mean()
    post_pca_centroid_g2_y = X_pca[y==1, 1].mean()

    ax.scatter(post_pca_centroid_g1_x, post_pca_centroid_g1_y, label = 'G1_centroid', color = 'blue')
    ax.scatter(post_pca_centroid_g2_x, post_pca_centroid_g2_y, label = 'G2_centroid', color = 'red')

    plt.legend()
    plt.savefig(filename_pca, format = 'pdf')
    print(f'PCA plot saved to: {filename_pca}')
    
    centroid_pre_g1 = np.array(pre_pca_centroid_g1_x, pre_pca_centroid_g1_y)
    centroid_pre_g2 = np.array(pre_pca_centroid_g2_x, pre_pca_centroid_g2_y)
    centroid_post_g1 = np.array(post_pca_centroid_g1_x, post_pca_centroid_g1_y)
    centroid_post_g2 = np.array(post_pca_centroid_g2_x, post_pca_centroid_g2_y)
    
    return {
        'pre_pca_g1': centroid_pre_g1,
        'pre_pca_g2': centroid_pre_g2,
        'post_pca_g1': centroid_post_g1,
        'post_pca_g2': centroid_post_g2,
    }

if __name__ == '__main__':
    N1 = 100
    d1 = 100
    N2 = 400
    d2 = 100
    filename_pca = './figs/pca.pdf'
    G1 = torch.randn(N1, d1) * 1
    G2 = torch.randn(N2, d2) * 3 + 2
    print(G1.shape, G2.shape)
    run_and_plot_pca(filename_pca, G1, G2)