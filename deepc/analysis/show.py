import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def show_embeddings_3d(image_tensor, labels_tensor, y_tensor):
    img, l, y = image_tensor.numpy(), labels_tensor.numpy(), y_tensor.detach().numpy().transpose((0, 2, 3, 1))
    ids = np.unique(l)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(90, 0)

    for i in ids:
        cluster_points = y[l == i]
        alpha = 0.1 if i == 0 else None
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], alpha=alpha)
