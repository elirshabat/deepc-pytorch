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
from deepc.datasets import augmentations, LimitedDataset, PrefetchedDataset
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.fit import fit
from deepc.run.checkpoints import load_checkpoints, create_checkpoints, show_checkpoints, update_checkpoints


GRADUAL_LEN_START = 100


def get_args():
    """
    Get the user arguments.
    :return: Arguments structure
    """
    parser = argparse.ArgumentParser(description="Run deepc training")

    parser.add_argument("paths_file", help="path to paths configuration file")
    parser.add_argument("checkpoints", help="path to checkpoints file")
    parser.add_argument("--dataset-name", "--dataset", choices=['coco2014', 'coco2017'], help="name of dataset to use")
    parser.add_argument("--arch", choices=['resnet'], help="the model to use")
    parser.add_argument("--out-dim", "-d", type=int, help="dimension of network outputs")
    parser.add_argument("--resize", "-r", type=int, nargs=2, help="tuple of (height, width) to resize the input images")
    parser.add_argument("--epochs", "--epoch-limit", "-e", type=int, default=float('inf'),
                        help="number of epochs to run")
    parser.add_argument("--batch-size", "-b", type=int, help="batch size to use")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="number of workers to use for reading data")
    parser.add_argument("--log-level", "--ll", default="INFO", choices=['DEBUG', 'INFO'], help="logging level")
    parser.add_argument("--lr", "--learning-rate", type=float, help="learning-rate")
    parser.add_argument("--pre-trained", action='store_true',
                        help="indicate whether or not to use pre-trained model in case not checkpoints were given")
    parser.add_argument("--save-freq", "-s", type=int, default=10, help="frequency in minutes for saving checkpoints")
    parser.add_argument("--log-freq", "-l", type=int, help="frequency in minutes for logging info")
    parser.add_argument("--profile", action='store_true', help="run single iteration with profiler")
    parser.add_argument("--dataset-limit", "--dl", type=int, help="limit the size of the dataset")
    parser.add_argument("--prefetch", action='store_true', help="load all dataset to ram upon initialization")
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


def create_dataset(data_dir, config_file, resize=None, len_limit=None, prefetch=False):
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
    if prefetch:
        train_set = PrefetchedDataset(train_set)
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
    if os.path.isfile(args.checkpoints):
        checkpoints = load_checkpoints(args.checkpoints)
        print(f"Loaded checkpoints from '{args.checkpoints}'")
        checkpoints = update_checkpoints(checkpoints, learning_rate=args.lr, batch_size=args.batch_size,
                                         dataset_name=args.dataset_name, arch=args.arch, out_dim=args.out_dim,
                                         resize=args.resize, dataset_limit=args.dataset_limit)
    else:
        checkpoints = create_checkpoints(dataset_name=args.dataset_name, arch=args.arch,
                                         out_dim=args.out_dim, resize=args.resize, batch_size=args.batch_size,
                                         learning_rate=args.lr, dataset_limit=args.dataset_limit)

    if checkpoints['batch_size'] is None:
        checkpoints['batch_size'] = 1
        warnings.warn(
            f"Batch size is set to the default value of {checkpoints['batch_size']} since it is the first training "
            f"and it is not passed as argument")

    if checkpoints['arch'] is None:
        checkpoints['arch'] = 'resnet'
        warnings.warn(f"Architecture name is not passed. Using default architecture '{checkpoints['arch']}'")

    if checkpoints['out_dim'] is None:
        checkpoints['out_dim'] = 5
        warnings.warn(f"Output dimension is not passed. Using default value: '{checkpoints['out_dim']}'")

    # Training set:
    default_dataset_name = 'coco2014'
    default_resize = [240, 320]

    dataset_name = checkpoints['dataset_name'] if checkpoints['dataset_name'] is not None else default_dataset_name
    resize = checkpoints['resize'] if checkpoints['resize'] is not None else default_resize

    train_set = create_dataset(paths[f"{dataset_name}_train_data"],
                               paths[f"{dataset_name}_train_config"],
                               resize=resize,
                               len_limit=checkpoints['dataset_limit'],
                               prefetch=args.prefetch)

    # Train data loader:
    train_loader = DataLoader(train_set, shuffle=True, num_workers=args.num_workers,
                              batch_size=checkpoints['batch_size'], pin_memory=cuda_available)

    # TODO: dev set and dev data loader

    # Model:
    model = create_model(checkpoints['arch'], checkpoints['out_dim'], device, parameters=checkpoints['model_params'],
                         pre_trained=args.pre_trained)

    # Loss function:
    loss_func = DiscriminativeLoss().to(device)

    # Optimizer:
    if checkpoints['learning_rate'] is None:
        raise ValueError("Learning-rate must be passed to first-time training")

    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoints['learning_rate'])

    # Training:
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
