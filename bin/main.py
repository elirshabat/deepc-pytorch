import os.path
import yaml
import torch
import matplotlib.pyplot as plt
import time
import argparse
import warnings
import logging
import sys
import multiprocessing as mp
from torchvision import transforms

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
    parser.add_argument("--epoch-limit", "-l", type=int, default=1e9, help="maximum number of epochs to run")
    parser.add_argument("--interactive", "-i", action='store_true',
                        help="whether or not to show debug info interactively")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="batch size to use")
    parser.add_argument("--num-workers", "-n", type=int, default=0, help="number of workers to use for reading data")
    parser.add_argument("--iter-size", "-t", type=int, help="iteration size for saving stats and parameters")
    parser.add_argument("--parameters", "-p", help="path to model's parameters file")
    parser.add_argument("--debug-level", "-d", default="INFO")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning-rate")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)

    mp.set_start_method('spawn')

    if args.interactive:
        plt.ion()

    logger = logging.getLogger('train')
    logger.setLevel(args.debug_level)
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
        stats_dir = paths['stats_dir']

    out_channels = args.out_dim
    image_height, image_width = args.resize[0], args.resize[1]

    parameters_file = args.parameters
    if parameters_file and not os.path.isfile(parameters_file):
        warnings.warn("Parameters file not found - creating new one")

    if args.model == 'resnet':
        if parameters_file:
            if os.path.isfile(parameters_file):
                model = ResnetMIS(pretrained_resnet=False, out_channels=out_channels)
                model.load_state_dict(torch.load(parameters_file, map_location='cpu'))
            else:
                model = ResnetMIS(pretrained_resnet=True, out_channels=out_channels)
        else:
            parameters_file = f"{args.model}_parameters-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"

    loss_func = DiscriminativeLoss()

    if torch.cuda.device_count() > 0:
        model = model.cuda()
    else:
        warnings.warn("Operating without GPU")

    composed_transforms = transforms.Compose([augmentations.Resize(image_height, image_width),
                                              augmentations.Normalize()])
    train_set = CocoDataset(train_anns_file, train_img_dir, transform=composed_transforms)
    dev_set = CocoDataset(dev_anns_file, dev_img_dir, transform=composed_transforms)

    train_stats_file_name = f"{args.model}_train_stats-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"
    def_stats_file_name = f"{args.model}_def_stats-out_dim_{out_channels}_h_{image_height}_w_{image_width}.pkl"
    train_stats_file = os.path.join(stats_dir, train_stats_file_name)
    dev_stats_file = os.path.join(stats_dir, def_stats_file_name)

    train_instance = Train(model, loss_func, train_set, dev_set=dev_set, params_path=parameters_file,
                           num_workers=args.num_workers, train_stats_path=train_stats_file,
                           dev_stats_path=dev_stats_file, iteration_size=args.iter_size, interactive=args.interactive,
                           batch_size=args.batch_size, learning_rate=args.lr)

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
