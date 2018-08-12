import argparse
import os.path
import warnings
import sys
import torch
from torch.utils.data import RandomSampler, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)
from deepc.modules.resnet import ResnetMIS
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
from deepc.analysis.show import embeddings_parallel_coordinates, show_outcomes
from deepc.analysis.metrics import kmeans


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", help="path to dataset file")
    parser.add_argument("data_path", help="path to data")
    parser.add_argument("--shuffle", action='store_true', help="whether or not to shuffle the data")
    parser.add_argument("--out-dim", '-d', type=int, default=5, help="dimension of network outputs")
    parser.add_argument("--resize", "-r", type=int, nargs=2, default=[240, 320],
                        help="tuple of (height, width) to resize the input images")
    parser.add_argument("--interactive", "-i", action='store_true',
                        help="whether or not to show debug info interactively")
    parser.add_argument("--parameters", "-p", help="path to model's parameters file")
    parser.add_argument("--num-samples", "-N", type=int, default=1, help="number of samples")
    args = parser.parse_args()

    parameters_file = args.parameters
    if parameters_file and not os.path.isfile(parameters_file):
        raise ValueError("Parameters file is not found")

    out_channels = args.out_dim

    if parameters_file:
        model = ResnetMIS(pretrained_resnet=False, out_channels=out_channels)
        model.load_state_dict(torch.load(parameters_file, map_location='cpu'))
    else:
        warnings.warn("Parameters file is not given - using non-trained model")
        model = ResnetMIS(pretrained_resnet=True, out_channels=out_channels)

    loss_func = DiscriminativeLoss()

    if torch.cuda.device_count() > 0:
        model = model.cuda()
    else:
        warnings.warn("Operating without GPU")

    composed_transforms = transforms.Compose([augmentations.Resize(args.resize[0], args.resize[1]),
                                              augmentations.Normalize()])
    dataset = CocoDataset(args.dataset_file, args.data_path, transform=composed_transforms)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler)

    with torch.no_grad():
        i = 0
        for sample in data_loader:
            if i > args.num_samples:
                break
            i += 1

            y = model(sample['image'].permute([0, 3, 1, 2]))
            embeddings_parallel_coordinates(y.squeeze(0), sample['labels'].squeeze(0))

            computed_clusters, cluster_centers = kmeans(y.squeeze(0).cpu().numpy(),
                                                        sample['labels'].squeeze(0).cpu().numpy())
            show_outcomes(sample['image'].squeeze(0), sample['labels'].squeeze(0), computed_clusters)

    plt.show()
