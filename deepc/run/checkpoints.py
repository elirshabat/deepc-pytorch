import torch
from deepc.analysis.show import show_learning_curve


def create_checkpoints(model=None, optimizer=None, train_learning_curve=[], dev_learning_curve=[], gradual_len=None):
    return {
        'model_params': model.state_dict() if model is not None else None,
        'optimizer_params': optimizer.state_dict() if optimizer is not None else None,
        'train_learning_curve': train_learning_curve,
        'dev_learning_curve': dev_learning_curve,
        'gradual_len': gradual_len
    }


def load_checkpoints(file_path):
    return torch.load(file_path, map_location='cpu')


def update_checkpoints(checkpoints, model=None, optimizer=None, train_learning_curve=[], dev_learning_curve=[],
                       gradual_len=None):
    if model is not None:
        checkpoints['model_params'] = model.state_dict()
    if optimizer is not None:
        checkpoints['optimizer_params'] = optimizer.state_dict()
    checkpoints['train_learning_curve'].extend(train_learning_curve)
    checkpoints['dev_learning_curve'].extend(dev_learning_curve)
    checkpoints['gradual_len'] = gradual_len
    return checkpoints


def save_checkpoints(checkpoints, file_path):
    torch.save(checkpoints, file_path)


def show_checkpoints(checkpoints):
    show_learning_curve(checkpoints['train_learning_curve'], 'train')
    show_learning_curve(checkpoints['dev_learning_curve'], 'dev')
