import os.path
import yaml
import torch
import time
import argparse
import warnings
import logging
import sys
import multiprocessing as mp
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import cProfile
import numpy as np
import matplotlib.pyplot as plt

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations, GradualDataset
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.fit import fit
from deepc.run.checkpoints import load_checkpoints, create_checkpoints, show_checkpoints


GRADUAL_LEN_START = 100


def get_args():
    """
    Get the user arguments.
    :return: Arguments structure
    """
    parser = argparse.ArgumentParser(description="Run deepc training")

    parser.add_argument("paths_file", help="path to paths configuration file")
    parser.add_argument("--dataset-name", "--dataset", default='coco2014', choices=['coco2014', 'coco2017'],
                        help="name of dataset to use")
    parser.add_argument("--arch", default="resnet", choices=['resnet'], help="the model to use")
    parser.add_argument("--out-dims", "-d", type=int, default=5, help="dimension of network outputs")
    parser.add_argument("--resize", "-r", type=int, nargs=2, default=[240, 320],
                        help="tuple of (height, width) to resize the input images")
    parser.add_argument("--epochs", "--epoch-limit", "-e", type=int, default=10000, help="number of epochs to run")
    parser.add_argument("--batch-size", "-b", type=int, help="batch size to use")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="number of workers to use for reading data")
    parser.add_argument("--checkpoints", "-c", help="path to checkpoints file")
    parser.add_argument("--log-level", "--ll", default="INFO", choices=['DEBUG', 'INFO'], help="logging level")
    parser.add_argument("--lr", "--learning-rate", type=float, help="learning-rate")
    parser.add_argument("--pre-trained", action='store_true',
                        help="indicate whether or not to use pre-trained model in case not checkpoints were given")
    parser.add_argument("--save-freq", "-s", type=int, default=10, help="frequency in minutes for saving checkpoints")
    parser.add_argument("--log-freq", "-l", action='store_true', help="frequency in minutes for logging info")
    parser.add_argument("--profile", action='store_true', help="run single iteration with profiler")
    parser.add_argument("--gradual", "-g", type=float,
                        help="gradually increasing the dataset. "
                             "The argument is the loss threshold below which the size of the dataset is doubled.")
    return parser.parse_args()


def create_logger(name, level, arch):
    """
    Create logger object.
    :param name: Name of the logger
    :param level: Level to log
    :param arch: Architecture
    :return: Logger
    """
    local_logger = logging.getLogger(name)
    local_logger.setLevel(level)
    handler = logging.FileHandler(f"{name}_{arch}.log")
    formatter = logging.Formatter(f"%(asctime)s : %(levelname)s : {arch} : %(message)s")
    handler.setFormatter(formatter)
    local_logger.addHandler(handler)
    streamer = logging.StreamHandler()
    local_logger.addHandler(streamer)
    return local_logger


def create_dataset(data_dir, config_file, resize=None, gradual_len=None):
    """
    Create data-set.
    :param data_dir: Directory with data
    :param config_file:  Dataset's configuration file
    :param resize: Resize vector for 2D images
    :param gradual_len: Length for gradual dataset
    :return: Dataset
    """
    dataset_transforms = []
    if resize:
        dataset_transforms.append(augmentations.Resize(resize[0], resize[1]))
    dataset_transforms.append(augmentations.Normalize())
    train_set = CocoDataset(config_file, data_dir, transform=transforms.Compose(dataset_transforms))
    if gradual_len:
        train_set = GradualDataset(train_set, gradual_len)
    return train_set


def create_model(arch, out_dims, device, parameters=None, pre_trained=False):
    """
    Create the model.
    :param arch: Architecture
    :param out_dims: Dimension of output points
    :param device: pytorch device to train on
    :param parameters: Model's parameters
    :param pre_trained: Whether or not to use a pre-trained model
    :return: Model
    """
    if arch == 'resnet':
        if parameters is not None:
            model = ResnetMIS(pretrained_resnet=False, out_channels=out_dims)
            model.load_state_dict(parameters)
        elif pre_trained:
            model = ResnetMIS(pretrained_resnet=True, out_channels=out_dims)
        else:
            model = ResnetMIS(pretrained_resnet=False, out_channels=out_dims)
    else:
        raise ValueError(f"{arch} is not a valid model")
    return model.to(device)


def generate_checkpoints_file_name(arch):
    """
    Generate the name of the checkpoints file.
    :param arch: Architecture
    :return: File name
    """
    return f"{arch}_checkpoints.pkl"


def main(args):
    """
    Main function of the script.
    :param args: User's arguments
    """
    # Initial configurations:
    mp.set_start_method('spawn')
    cuda_available = (torch.cuda.device_count() > 0)
    device = torch.device('cuda') if cuda_available else torch.device('cpu')
    if cuda_available:
        cudnn.benchmark = True
    else:
        warnings.warn("Operating without GPU")
    logger = create_logger('train', args.log_level, args.arch)
    with open(args.paths_file, 'r') as f:
        paths = yaml.load(f)

    # Checkpoints:
    if not args.checkpoints:
        args.checkpoints = generate_checkpoints_file_name(args.arch)
        if os.path.isfile(args.checkpoints):
            warnings.warn(f"Using existing checkpoints file with default filename: '{args.checkpoints}'")
        else:
            warnings.warn(f"Using new checkpoints file with default filename: '{args.checkpoints}'")
    if os.path.isfile(args.checkpoints):
        checkpoints = load_checkpoints(args.checkpoints)
        print(f"Loaded checkpoints from '{args.checkpoints}'")
    else:
        checkpoints = create_checkpoints()

    # Training set:
    train_set = create_dataset(paths[f'{args.dataset_name}_train_data'], paths[f'{args.dataset_name}_train_config'],
                               resize=args.resize, gradual_len=checkpoints['gradual_len'])

    # Train data loader:
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif checkpoints['batch_size'] is not None:
        batch_size = checkpoints['batch_size']
    else:
        batch_size = 1
        warnings.warn(f"Batch size is set to the default value of {batch_size} since it is the first training and it "
                      f"is not passed as argument")
    train_loader = DataLoader(train_set, shuffle=True, num_workers=args.num_workers,
                              batch_size=batch_size, pin_memory=cuda_available)

    # TODO: dev set and dev data loader

    # TODO: enable gradual dataset

    # Model:
    model = create_model(args.arch, args.out_dims, device, parameters=checkpoints['model_params'],
                         pre_trained=args.pre_trained)

    # Loss function:
    loss_func = DiscriminativeLoss().to(device)

    # Optimizer:
    if args.lr is not None:
        lr = args.lr
    elif checkpoints['learning_rate'] is not None:
        lr = checkpoints['learning_rate']
    else:
        lr = 1e-3
        warnings.warn(f"Learning rate is set to the default value of {lr} since it is the first training and it is "
                      f"not passed as argument")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if checkpoints['optimizer_params'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_params'])

    start_training_time = time.time()

    loss_curve = fit(model, loss_func, optimizer, train_loader, args.epochs, device, logger,
                     start_epoch=len(checkpoints['train_learning_curve']), checkpoints=checkpoints,
                     checkpoints_file_path=args.checkpoints, save_freq=args.save_freq, log_freq=args.log_freq)

    logger.info(f"Done training - n_epochs:{args.epochs} time:{time.time() - start_training_time} "
                f"avg-loss:{np.mean(loss_curve)}")

    # TODO: validation

    show_checkpoints(checkpoints)
    plt.show()


if __name__ == '__main__':
    arguments = get_args()
    if arguments.profile:
        cProfile.run('main(arguments)', 'main_profile')
    else:
        main(arguments)
