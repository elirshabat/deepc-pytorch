from torch.utils.data import DataLoader
import numpy as np


def calc_dataset_metrics(dataset, num_workers=0):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers)
    num_instances = [len(sample['labels'].unique()) for sample in data_loader]
    np_num_instances = np.sort(np.array(num_instances))

    instance_metrics = dict()
    instance_metrics['mean'] = np.mean(np_num_instances)
    instance_metrics['std'] = np.std(np_num_instances)
    instance_metrics['max'] = np.max(np_num_instances)
    instance_metrics['min'] = np.min(np_num_instances)
    instance_metrics['median'] = np_num_instances[len(np_num_instances)//2]
    instance_metrics['median095'] = np_num_instances[int(len(np_num_instances)*0.95)]

    return instance_metrics
