import argparse
import os.path
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert parameters file to checkpoints file")
    parser.add_argument("parameters_file", help="path to parameters file")
    parser.add_argument("checkpoints_file", help="path to parameters file")
    args = parser.parse_args()

    if not os.path.isfile(args.parameters_file):
        raise ValueError("Parameters file does not exist")

    checkpoints = {
        'train_epoch': 0,
        'train_iteration': 0,
        'dev_epoch': 0,
        'dev_iteration': 0,
        'epochs_loss_curve': [],
        'iteration_loss_curve': [],
        'model_params': torch.load(args.parameters_file, map_location='cpu'),
        'optimizer_params': None
    }

    torch.save(checkpoints, args.checkpoints_file)
    print("Successfully converted parameters file to checkpoints file")
