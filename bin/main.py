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
import itertools

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.train import Train
from deepc.analysis import AverageMeter


def get_args():
    parser = argparse.ArgumentParser(description="Run deepc training")

    parser.add_argument("paths_file", help="path to paths configuration file")
    parser.add_argument("--dataset-name", "--dataset", default='coco2014', choices=['coco2014', 'coco2017'],
                        help="name of dataset to use")
    parser.add_argument("--arch", default="resnet", choices=['resnet'], help="the model to use")
    parser.add_argument("--out-dims", "-d", type=int, default=5, help="dimension of network outputs")
    parser.add_argument("--resize", "-r", type=int, nargs=2, default=[240, 320],
                        help="tuple of (height, width) to resize the input images")
    parser.add_argument("--epochs", "--epoch-limit", "-e", type=int, default=1e9,
                        help="maximum number of epochs to run")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size to use")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="number of workers to use for reading data")
    parser.add_argument("--iter-size", "-T", type=int, default=100,
                        help="iteration size for saving checkpoints")
    parser.add_argument("--checkpoints", "-c", help="path to checkpoints file")
    parser.add_argument("--log-level", "--ll", default="INFO", choices=['DEBUG', 'INFO'], help="logging level")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4, help="learning-rate")
    parser.add_argument("--no-dev", action="store_true", help="train without dev-set")
    parser.add_argument("--pre-trained", action='store_true',
                        help="indicate whether or not to use pre-trained model in case not checkpoints were given")
    parser.add_argument("--save-freq", "-s", type=int, default=0,
                        help="frequency in iterations for saving checkpoints (0 means every epoch)")

    return parser.parse_args()


def create_logger(name, level, arch):
    local_logger = logging.getLogger(name)
    local_logger.setLevel(level)
    handler = logging.FileHandler(f"{name}_{arch}.log")
    formatter = logging.Formatter(f"%(asctime)s : %(levelname)s : {arch} : %(message)s")
    handler.setFormatter(formatter)
    local_logger.addHandler(handler)
    streamer = logging.StreamHandler()
    local_logger.addHandler(streamer)
    return local_logger


if __name__ == '__main__':

    args = get_args()
    print(args)

    # Initial configurations:
    mp.set_start_method('spawn')
    cuda_available = (torch.cuda.device_count() > 0)
    if cuda_available:
        cudnn.benchmark = True
    else:
        warnings.warn("Operating without GPU")
    logger = create_logger('train', args.log_level, args.arch)
    with open(args.paths_file, 'r') as f:
        paths = yaml.load(f)

    # Create train dataset:
    train_data_dir = paths[f'{args.dataset_name}_train_data']
    train_dataset_file = paths[f'{args.dataset_name}_train_config']
    train_set_transforms = []
    if args.resize:
        train_set_transforms.append(augmentations.Resize(args.resize[0], args.resize[1]))
    train_set_transforms.append(augmentations.Normalize())
    train_set = CocoDataset(train_dataset_file, train_data_dir, transform=transforms.Compose(train_set_transforms))

    # Create dev dataset:
    if not args.no_dev:
        dev_data_dir = paths[f'{args.dataset_name}_dev_data']
        dev_dataset_file = paths[f'{args.dataset_name}_dev_config']
        dev_set_transforms = []
        if args.resize:
            dev_set_transforms.append(augmentations.Resize(args.resize[0], args.resize[1]))
        dev_set_transforms.append(augmentations.Normalize())
        dev_set = CocoDataset(dev_dataset_file, dev_data_dir, transform=transforms.Compose(dev_set_transforms))
    else:
        warnings.warn("Training without dev set")
        dev_set = None

    # Load checkpoints:
    if not args.checkpoints:
        args.checkpoints = f"{args.arch}_checkpoints.pkl"
        if os.path.isfile(args.checkpoints):
            warnings.warn(f"Using existing checkpoints file with default filename: '{args.checkpoints}'")
        else:
            warnings.warn(f"Using new checkpoints file with default filename: '{args.checkpoints}'")
    if os.path.isfile(args.checkpoints):
        checkpoints = torch.load(args.checkpoints, map_location='cpu')
        print(f"Loaded checkpoints from '{args.checkpoints}'")
    else:
        checkpoints = {
            'train_epoch': 0,
            'train_iteration': 0,
            'dev_epoch': 0,
            'dev_iteration': 0,
            'epochs_loss_curve': [],
            'iteration_loss_curve': [],
            'model_params': None,
            'optimizer_params': None
        }

    # Create the model:
    if args.arch == 'resnet':
        if checkpoints['model_params'] is not None:
            model = ResnetMIS(pretrained_resnet=False, out_channels=args.out_dims)
            model.load_state_dict(checkpoints['model_params'])
        elif args.pre_trained:
            model = ResnetMIS(pretrained_resnet=True, out_channels=args.out_dims)
        else:
            model = ResnetMIS(pretrained_resnet=False, out_channels=args.out_dims)
    if cuda_available:
        model = model.cuda()

    # Create loss function:
    loss_func = DiscriminativeLoss()
    if cuda_available:
        loss_func = loss_func.cuda()

    # Create optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if checkpoints['optimizer_params'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_params'])

    # Create data loaders:
    train_loader = DataLoader(train_set, shuffle=True, num_workers=args.num_workers,
                              batch_size=args.batch_size, pin_memory=cuda_available)
    if dev_set is not None:
        dev_loader = DataLoader(dev_set, shuffle=True, num_workers=args.num_workers,
                                batch_size=args.batch_size, pin_memory=cuda_available)
    else:
        dev_loader = None

    # Run training:
    start_train_epoch, start_train_iteration = checkpoints['train_epoch'], checkpoints['train_iteration']

    train_instance = Train(train_loader, model, loss_func, optimizer, cpu=(not cuda_available),
                           start_epoch=start_train_epoch, start_iteration=start_train_iteration,
                           iteration_size=args.iter_size)

    train_epoch_len = len(train_set)//(args.batch_size*args.iter_size)

    start_training_time = time.time()
    epoch_losses = []

    steps_counter = itertools.count(1)

    while train_instance.epoch < start_train_epoch + args.epochs:
        step = next(steps_counter)

        iteration_loss, epoch_done, avg_times = train_instance.run()

        checkpoints['train_iteration'] += 1
        checkpoints['train_epoch'] = train_instance.epoch
        checkpoints['iteration_loss_curve'].append(iteration_loss)
        checkpoints['model_params'] = model.state_dict()
        checkpoints['optimizer_params'] = optimizer.state_dict()

        epoch_losses.append(iteration_loss)

        logger.info(f"Train iteration - epoch:{train_instance.epoch} "
                    f"iteration:{train_instance.iteration} avg-loss:{iteration_loss}")
        logger.debug("Iteration times - " + " ".join([f"{key}:{avg_times[key]}" for key in avg_times]))

        if epoch_done:
            epoch_avg_loss = sum(epoch_losses)/len(epoch_losses)
            checkpoints['epochs_loss_curve'].append(epoch_avg_loss)
            torch.save(checkpoints, args.checkpoints)
            logger.info(f"Train epoch {train_instance.epoch} - avg-loss:{epoch_avg_loss}")
            epoch_losses = []
        else:
            if args.save_freq and (step % args.save_freq) == 0:
                torch.save(checkpoints, args.checkpoints)

    logger.info(f"Done training - n_epochs:{args.epochs} time:{time.time() - start_training_time}")

    # TODO: validation

    # TODO: show curves in case of interactive mode
