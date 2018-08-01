import os.path
import sys
import yaml
import torch
import matplotlib.pyplot as plt
import time

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
        train_img_dir = paths['coco_train2014_images']
        train_anns_file = paths['coco_train2014_annotations']
        dev_img_dir = paths['coco_dev2014_images']
        dev_anns_file = paths['coco_dev2014_annotations']

    out_channels = 3
    parameters_file = os.path.join(repo_dir, "Test", "parameters", "limited_resnet_parameters.pkl")
    model = ResnetMIS(pretrained_resnet=True, out_channels=out_channels)
    if os.path.isfile(parameters_file):
        model.load_state_dict(torch.load(parameters_file))

    loss_func = DiscriminativeLoss(out_channels)

    if torch.cuda.device_count() > 0:
        model = model.cuda()
    else:
        print("Warning: operating without GPU")

    image_height, image_width = 240//2, 320//2

    train_set_size = 5
    train_set = LimitedDataset(train_set_size, train_anns_file, train_img_dir,
                               augmentations.Resize(image_height, image_width))
    dev_set_size = 2
    dev_set = LimitedDataset(dev_set_size, dev_anns_file, dev_img_dir,
                             augmentations.Resize(image_height, image_width))

    train_stats_file = os.path.join(repo_dir, "Test", "parameters", "limited_resnet_stats_train.pkl")
    dev_stats_file = os.path.join(repo_dir, "Test", "parameters", "limited_resnet_stats_dev.pkl")

    train_instance = Train(model, loss_func, train_set, dev_set=dev_set, params_path=parameters_file, num_workers=0,
                           train_stats_path=train_stats_file, dev_stats_path=dev_stats_file)

    start_time = time.time()
    num_epochs = 10
    train_instance.run(max_epochs=num_epochs)
    end_time = time.time()

    print(f"Execution time for {num_epochs} epoches: {end_time - start_time}")

    train_stats = analysis.load(train_stats_file)
    train_stats.plot()

    dev_stats = analysis.load(dev_stats_file)
    dev_stats.plot()

    plt.show()
