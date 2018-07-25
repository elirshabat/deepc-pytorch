import os.path
import sys
import yaml
import torch

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.train import Train
from deepc.analysis import analysis


class LimitedDataset(CocoDataset):

    def __init__(self, limit, anns_file, images_dir, transform=None):
        super().__init__(anns_file, images_dir, transform=transform)

        self._limit = limit

    def __len__(self):
        return self._limit


if __name__ == '__main__':

    local_dir = os.path.join(repo_dir, "Local")
    paths_config_file_path = os.path.join(local_dir, "paths.yaml")
    with open(paths_config_file_path, 'r') as f:
        paths = yaml.load(f)
        img_dir = paths['coco_train2014_images']
        anns_file = paths['coco_train2014_annotations']

    parameters_file = os.path.join(repo_dir, "Test", "parameters", "limited_resnet_parameters.pkl")

    model = ResnetMIS(pretrained_resnet=True)

    if os.path.isfile(parameters_file):
        model.load_state_dict(torch.load(parameters_file))
    loss_func = DiscriminativeLoss(1, 1, 2, 2, 1)

    limit = 2
    dataset = LimitedDataset(limit, anns_file, img_dir, augmentations.Resize(240//2, 320//2))

    stats_file = os.path.join(repo_dir, "Test", "parameters", "limited_resnet_stats.pkl")
    train_instance = Train(model, loss_func, dataset, params_path=parameters_file, num_workers=0, train_stats_path=stats_file)
    train_instance.run(max_epochs=10)

    stats = analysis.load(stats_file)
    stats.show()
