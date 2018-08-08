import numpy as np


def show_distance_matrix(image_tensor, labels_tensor, y_tensor):
    img, l, y = image_tensor.numpy(), labels_tensor.numpy(), y_tensor.detach().numpy()
    ids = np.unique(l)

    for i in ids:
        cluster_points = img[l == i]
        center = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - center, 2, axis=1)
        # TODO: continue
