from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans


def get_image_metrics(dataset, num_workers):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers)
    mean_img = np.zeros(dataset[0]['image'].shape, dtype=np.float64)
    var_img = np.zeros(dataset[0]['image'].shape, dtype=np.float64)
    n = 0.0
    for s in data_loader:
        n += 1.0
        mean_img = (1.0/n)*((n - 1)*mean_img + s['image'].numpy())
    n = 0.0
    for s in data_loader:
        n += 1.0
        var_img = (1.0/n)*((n - 1)*var_img + (s['image'].numpy() - mean_img)**2)
    return mean_img, var_img


def calc_dataset_metrics(dataset, num_workers=0):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers)

    mean_img, var_img = get_image_metrics(dataset, num_workers)
    image_metrics = {
        'mean': mean_img,
        'var': var_img
    }

    num_instances = [len(sample['labels'].unique()) for sample in data_loader]
    np_num_instances = np.sort(np.array(num_instances))

    instance_metrics = dict()
    instance_metrics['mean'] = np.mean(np_num_instances)
    instance_metrics['std'] = np.std(np_num_instances)
    instance_metrics['max'] = np.max(np_num_instances)
    instance_metrics['min'] = np.min(np_num_instances)
    instance_metrics['median'] = np_num_instances[len(np_num_instances)//2]
    instance_metrics['median095'] = np_num_instances[int(len(np_num_instances)*0.95)]

    return instance_metrics, image_metrics


def kmeans(x, labels):
    x = x.transpose((1, 2, 0))
    true_labels, x = labels.reshape(-1), x.reshape((-1, x.shape[2]))
    n_clusters = len(np.unique(true_labels))
    y = KMeans(n_clusters=n_clusters, n_init=20, precompute_distances=True).fit(x)
    return y.labels_, y.cluster_centers_
