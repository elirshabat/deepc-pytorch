import os.path
import yaml
import torch
import matplotlib.pyplot as plt
import time
import argparse
import warnings
import logging
import sys

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.train import Train
from deepc.analysis import analysis


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("paths_file", help="path to paths configuration file")
    parser.add_argument("--out-dim", type=int, default=3, help="dimension of network outputs")
    parser.add_argument("--model", default="resnet", help="the model to use")
    parser.add_argument("--resize", "-r", type=int, nargs=2, default=[240, 320],
                        help="tuple of (height, width) to resize the input images")
    parser.add_argument("--epoch-limit", "-l", type=int, default=1e9)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('train.log')
    formatter = logging.Formatter(f"%(asctime)s : %(levelname)s : {args.model} : %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    streamer = logging.StreamHandler()
    logger.addHandler(streamer)

    with open(args.paths_file, 'r') as f:
        paths = yaml.load(f)
        train_img_dir = paths['coco_train2014_images']
        train_anns_file = paths['coco_train2014_annotations']
        dev_img_dir = paths['coco_dev2014_images']
        dev_anns_file = paths['coco_dev2014_annotations']
        params_dir = paths['parameters_dir']
        stats_dir = paths['stats_dir']

    out_channels = args.out_dim
    image_height, image_width = args.resize[0], args.resize[1]

    params_file_name = f"{args.model}_parameters-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"
    parameters_file = os.path.join(params_dir, params_file_name)

    if args.model == 'resnet':
        if os.path.isfile(parameters_file):
            model = ResnetMIS(pretrained_resnet=False, out_channels=out_channels)
            model.load_state_dict(torch.load(parameters_file))
        else:
            model = ResnetMIS(pretrained_resnet=True, out_channels=out_channels)

    loss_func = DiscriminativeLoss(out_channels)

    if torch.cuda.device_count() > 0:
        model = model.cuda()
    else:
        warnings.warn("Operating without GPU")

    train_set = CocoDataset(train_anns_file, train_img_dir, augmentations.Resize(image_height, image_width))
    dev_set = CocoDataset(dev_anns_file, dev_img_dir, augmentations.Resize(image_height, image_width))

    train_stats_file_name = f"{args.model}_train_stats-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"
    def_stats_file_name = f"{args.model}_def_stats-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"
    train_stats_file = os.path.join(stats_dir, train_stats_file_name)
    dev_stats_file = os.path.join(stats_dir, def_stats_file_name)

    train_instance = Train(model, loss_func, train_set, dev_set=dev_set, params_path=parameters_file, num_workers=0,
                           train_stats_path=train_stats_file, dev_stats_path=dev_stats_file, iteration_size=5)

    start_time = time.time()
    num_epochs = args.epoch_limit
    train_instance.run(max_epochs=num_epochs)
    end_time = time.time()

    print(f"Execution time for {num_epochs} epoches: {end_time - start_time}")

    train_stats = analysis.load(train_stats_file)
    train_stats.plot()

    dev_stats = analysis.load(dev_stats_file)
    dev_stats.plot()

    plt.show(block=True)
