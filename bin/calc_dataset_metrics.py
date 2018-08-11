import argparse
import os.path
import sys

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
from deepc.analysis import metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("dataset_file", help="path to annotations file")
    parser.add_argument("data_path", help="path to data location")
    parser.add_argument("--resize", "-r", type=int, nargs=2, default=[240, 320],
                        help="tuple of (height, width) to resize the input images")
    parser.add_argument("--num-workers", "-n", type=int, default=0, help="number of workers to use for reading data")

    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    if dataset_name == 'coco':
        dataset = CocoDataset(args.dataset_file, args.data_path, augmentations.Resize(args.resize[0], args.resize[1]))
    else:
        raise ValueError(f"Unknown dataset - {args.dataset}")

    instance_metrics = metrics.calc_dataset_metrics(dataset, args.num_workers)

    print(f"Instance metrics:\n{instance_metrics}")
