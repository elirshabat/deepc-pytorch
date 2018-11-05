import argparse
import os.path
import sys
from torchvision import transforms
import pickle

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations, LimitedDataset
from deepc.analysis import metrics


def create_dataset(data_dir, config_file, resize=None, len_limit=None):
    """
    Create data-set.
    :param data_dir: Directory with data
    :param config_file:  Dataset's configuration file
    :param resize: Resize vector for 2D images
    :param len_limit: Length for gradual dataset
    :return: Dataset
    """
    dataset_transforms = []
    if resize:
        dataset_transforms.append(augmentations.Resize(resize[0], resize[1]))
    dataset_transforms.append(augmentations.Normalize())
    train_set = CocoDataset(config_file, data_dir, transform=transforms.Compose(dataset_transforms))
    if len_limit:
        train_set = LimitedDataset(train_set, len_limit)
    return train_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to dataset configuration file")
    parser.add_argument("data_dir", help="path to data directory")
    parser.add_argument("--resize", "-r", type=int, nargs=2, help="tuple of (height, width) to resize the input images")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="number of workers to use for reading data")
    parser.add_argument("--dataset-limit", "--dl", type=int, help="limit the size of the dataset")
    parser.add_argument("--output-dir", "-o", help="path to output directory")
    args = parser.parse_args()

    print(args)

    dataset = create_dataset(args.data_dir, args.config_file, resize=args.resize, len_limit=args.dataset_limit)
    instance_metrics, image_metrics = metrics.calc_dataset_metrics(dataset, args.num_workers)

    instance_metrics_file_name = 'instance_metrics.pkl'
    instance_metrics_file_path = os.path.join(args.output_dir, instance_metrics_file_name) \
        if args.output_dir is not None else instance_metrics_file_name
    image_metrics_file_name = 'image_metrics.pkl'
    image_metrics_file_path = os.path.join(args.output_dir, image_metrics_file_name) \
        if args.output_dir is not None else image_metrics_file_name

    with open(instance_metrics_file_path, 'wb') as f:
        pickle.dump(instance_metrics, f)
    with open(image_metrics_file_path, 'wb') as f:
        pickle.dump(image_metrics, f)

    print(f"Instance metrics:\n{instance_metrics}")
