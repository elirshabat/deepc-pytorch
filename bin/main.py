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

curr_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.path.join(curr_dir, "..")
sys.path.append(repo_dir)

from deepc.modules.resnet import ResnetMIS
from deepc.datasets.coco import CocoDataset
from deepc.datasets import augmentations, GradualDataset
from deepc.loss.discriminative import DiscriminativeLoss
from deepc.run.train import Train
from deepc.run.fit import fit
from deepc.run.checkpoints import load_checkpoints, create_checkpoints


GRADUAL_LEN_START = 100


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
    parser.add_argument("--save-freq", "-s", type=int, default=10,
                        help="frequency in minutes for saving checkpoints")
    parser.add_argument("--profile", action='store_true', help="run single iteration with profiler")
    parser.add_argument("--gradual", "-g", type=float,
                        help="gradually increasing the dataset. "
                             "The argument is the loss threshold after which the size of the dataset is doubled.")

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


# def load_checkpoints(file_path, arch, gradual_len_start):
#     """
#     Load checkpoints from file or create new checkpoints object.
#     :param file_path: Path to checkpoints file (either existing or not)
#     :param arch: Architecture name to use in the file name in case it is not given.
#     :param gradual_len_start: First number of samples to use in case of training with gradual dataset.
#     :return: Checkpoints object.
#     """
#     if not file_path:
#         file_path = f"{arch}_checkpoints.pkl"
#         if os.path.isfile(file_path):
#             warnings.warn(f"Using existing checkpoints file with default filename: '{file_path}'")
#         else:
#             warnings.warn(f"Using new checkpoints file with default filename: '{file_path}'")
#
#     if os.path.isfile(file_path):
#         checkpoints = torch.load(file_path, map_location='cpu')
#         print(f"Loaded checkpoints from '{file_path}'")
#     else:
#         checkpoints = {
#             'train_epoch': 0,
#             'train_iteration': 0,
#             'dev_epoch': 0,
#             'dev_iteration': 0,
#             'epochs_loss_curve': [],
#             'iteration_loss_curve': [],
#             'model_params': None,
#             'optimizer_params': None,
#             'gradual_len': gradual_len_start
#         }
#
#     if 'gradual_len' not in checkpoints:
#         checkpoints['gradual_len'] = gradual_len_start
#         warnings.warn(f"Updated checkpoints - add 'gradual_len' field with value {checkpoints['gradual_len']}")
#
#     return checkpoints


def create_dataset(data_dir, config_file, resize=None, gradual_len=None):
    dataset_transforms = []
    if resize:
        dataset_transforms.append(augmentations.Resize(resize[0], resize[1]))
    dataset_transforms.append(augmentations.Normalize())
    train_set = CocoDataset(config_file, data_dir, transform=transforms.Compose(dataset_transforms))
    if gradual_len:
        train_set = GradualDataset(train_set, gradual_len)
    return train_set


def create_model(arch, out_dims, parameters=None, pre_trained=False, cuda_available=False):
    model = None
    if arch == 'resnet':
        if parameters is not None:
            model = ResnetMIS(pretrained_resnet=False, out_channels=out_dims)
            model.load_state_dict(parameters)
        elif pre_trained:
            model = ResnetMIS(pretrained_resnet=True, out_channels=out_dims)
        else:
            model = ResnetMIS(pretrained_resnet=False, out_channels=out_dims)
    if cuda_available:
        model = model.cuda()
    return model


def main(args):

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

    if not args.checkpoints:
        args.checkpoints = f"{arch}_checkpoints.pkl"
        if os.path.isfile(args.checkpoints):
            warnings.warn(f"Using existing checkpoints file with default filename: '{args.checkpoints}'")
        else:
            warnings.warn(f"Using new checkpoints file with default filename: '{args.checkpoints}'")

    if os.path.isfile(args.checkpoints):
        checkpoints = load_checkpoints(args.checkpoints)
        # checkpoints = torch.load(checkpoints_file_path, map_location='cpu')
        print(f"Loaded checkpoints from '{args.checkpoints}'")
    else:
        checkpoints = create_checkpoints()
        # checkpoints = {
        #     'train_epoch': 0,
        #     'train_iteration': 0,
        #     'dev_epoch': 0,
        #     'dev_iteration': 0,
        #     'epochs_loss_curve': [],
        #     'iteration_loss_curve': [],
        #     'model_params': None,
        #     'optimizer_params': None,
        #     'gradual_len': gradual_len_start
        # }

    # if 'gradual_len' not in checkpoints:
    #     checkpoints['gradual_len'] = gradual_len_start
    #     warnings.warn(f"Updated checkpoints - add 'gradual_len' field with value {checkpoints['gradual_len']}")

    # return checkpoints
    # checkpoints = load_checkpoints(args.checkpoints, args.arch, GRADUAL_LEN_START)

    train_set = create_dataset(paths[f'{args.dataset_name}_train_data'], paths[f'{args.dataset_name}_train_config'],
                               resize=args.resize, gradual_len=checkpoints['gradual_len'])

    # TODO: dev set
    # if not args.no_dev:
    #     dev_set = create_dataset(f'{args.dataset_name}_dev_data', f'{args.dataset_name}_dev_config', resize=args.resize)
    # else:
    #     warnings.warn("Training without dev set")
    #     dev_set = None

    model = create_model(args.arch, args.out_dims, parameters=checkpoints['model_params'], pre_trained=args.pre_trained,
                         cuda_available=cuda_available)

    loss_func = DiscriminativeLoss(cuda=cuda_available)
    if cuda_available:
        loss_func = loss_func.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if checkpoints['optimizer_params'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_params'])

    train_loader = DataLoader(train_set, shuffle=True, num_workers=args.num_workers,
                              batch_size=args.batch_size, pin_memory=cuda_available)

    # TODO: dev set
    # if dev_set is not None:
    #     dev_loader = DataLoader(dev_set, shuffle=True, num_workers=args.num_workers,
    #                             batch_size=args.batch_size, pin_memory=cuda_available)
    # else:
    #     dev_loader = None

    # Run training:
    # start_train_epoch, start_train_iteration = checkpoints['train_epoch'], checkpoints['train_iteration']

    # train_instance = Train(train_loader, model, loss_func, optimizer, cpu=(not cuda_available),
    #                        start_epoch=start_train_epoch, start_iteration=start_train_iteration,
    #                        iteration_size=args.iter_size)
    #
    # start_training_time = time.time()
    # epoch_losses = []
    #
    # last_save_time = time.time()

    fit(model, loss_func, optimizer, train_loader, args.epochs, device, logger,
        start_epoch=len(checkpoints['train_learning_curve']), checkpoints=checkpoints,
        checkpoints_file_path=args.checkpoints, save_freq=args.save_freq)

    # TODO: enable gradual dataset

    # while train_instance.epoch < start_train_epoch + args.epochs:

        # iteration_loss, epoch_done, avg_times = train_instance.run()

        # checkpoints['train_iteration'] += 1
        # checkpoints['train_epoch'] = train_instance.epoch
        # checkpoints['iteration_loss_curve'].append(iteration_loss)
        # checkpoints['model_params'] = model.state_dict()
        # checkpoints['optimizer_params'] = optimizer.state_dict()

        # epoch_losses.append(iteration_loss)

        # logger.info(f"Train iteration - epoch:{train_instance.epoch} "
        #             f"iteration:{train_instance.iteration} avg-loss:{iteration_loss} train-set-length:{len(train_set)}")
        # logger.debug("Iteration times - " + " ".join([f"{key}:{avg_times[key]}" for key in avg_times]))

        # if epoch_done:
        #     epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        #     checkpoints['epochs_loss_curve'].append(epoch_avg_loss)
        #     logger.info(f"Train epoch {train_instance.epoch} - avg-loss:{epoch_avg_loss}")
        #     epoch_losses = []
        #     if args.gradual and epoch_avg_loss <= args.gradual:
        #         train_set.len = train_set.len * 2

        # if (time.time() - last_save_time) / 60.0 < args.save_freq:
        #     torch.save(checkpoints, args.checkpoints)
        #     last_save_time = time.time()

    # logger.info(f"Done training - n_epochs:{args.epochs} time:{time.time() - start_training_time}")

    # TODO: validation

    # TODO: show curves in case of interactive mode


if __name__ == '__main__':

    arguments = get_args()
    print(arguments)

    if arguments.profile:
        cProfile.run('main(arguments)', 'main_profile')
    else:
        main(arguments)
