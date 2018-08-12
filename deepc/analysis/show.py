import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas
from deepc.datasets.augmentations.normalize import Normalize


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


def embeddings_parallel_coordinates(y_tensor, labels_tensor):
    l, y = labels_tensor.numpy(), y_tensor.detach().numpy().transpose((1, 2, 0))
    l, y = l.reshape(-1), y.reshape((-1, y.shape[2]))

    n_dims = y.shape[1]

    data = pandas.DataFrame(y, columns=[f"x{i}" for i in range(n_dims)])
    data['label'] = l

    plt.figure()
    pandas.plotting.parallel_coordinates(data, 'label')


def show_sample_data(image, labels):
    plt.figure()
    if image is not None:
        plt.imshow(image.cpu())
    if labels is not None:
        plt.imshow(labels.cpu(), alpha=0.5)


def show_samples_batch(sample, ignore_image=False, ignore_labels=False, is_normalized=False):
    if is_normalized:
        transform = Normalize(reverse=True)
        sample = transform(sample)
    image_batch, labels_batch = sample['image'], sample['labels']
    batch_size = image_batch.shape[0]
    for i in range(batch_size):
        image = None if ignore_image else image_batch[i, :, :, :]
        labels = None if ignore_labels else labels_batch[i, :, :]
        show_sample_data(image, labels)


def show_outcomes(image_tensor, true_labels_tensor, computed_labels_array):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    np_image = image_tensor.cpu().numpy()*255
    np_image = np_image.astype(np.uint8)
    ax1.imshow(np_image)
    ax2.imshow(np_image)
    ax3.imshow(np_image)
    ax2.imshow(true_labels_tensor.cpu().numpy(), alpha=0.5)
    ax3.imshow(np.reshape(computed_labels_array.astype(np.uint8), true_labels_tensor.cpu().numpy().shape), alpha=0.5)
