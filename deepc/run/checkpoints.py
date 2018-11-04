import torch
from deepc.analysis.show import show_learning_curve


def create_checkpoints(model=None, train_learning_curve=[], dev_learning_curve=[],
                       learning_rate=None, batch_size=None,
                       dataset_name=None, arch=None, out_dim=None, resize=None, dataset_limit=None):
    return {
        'arch': arch,
        'out_dim': out_dim,
        'dataset_name': dataset_name,
        'dataset_limit': dataset_limit,
        'resize': resize,
        'model_params': model.state_dict() if model is not None else None,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_learning_curve': train_learning_curve,
        'dev_learning_curve': dev_learning_curve
    }


def load_checkpoints(file_path):
    return torch.load(file_path, map_location='cpu')


def update_checkpoints(checkpoints, model=None, train_learning_curve=[], dev_learning_curve=[],
                       learning_rate=None, batch_size=None,
                       dataset_name=None, arch=None, out_dim=None, resize=None, dataset_limit=None):
    if model is not None:
        checkpoints['model_params'] = model.state_dict()
    checkpoints['train_learning_curve'].extend(train_learning_curve)
    checkpoints['dev_learning_curve'].extend(dev_learning_curve)
    if learning_rate is not None:
        checkpoints['learning_rate'] = learning_rate
    if batch_size is not None:
        checkpoints['batch_size'] = batch_size
    if dataset_name is not None:
        checkpoints['dataset_name'] = dataset_name
    if arch is not None:
        checkpoints['arch'] = arch
    if out_dim is not None:
        checkpoints['out_dim'] = out_dim
    if resize is not None:
        checkpoints['resize'] = resize
    if dataset_limit is not None:
        checkpoints['dataset_limit'] = dataset_limit
    return checkpoints


def save_checkpoints(checkpoints, file_path):
    torch.save(checkpoints, file_path)


def show_checkpoints(checkpoints):
    show_learning_curve(checkpoints['train_learning_curve'], 'train')
    show_learning_curve(checkpoints['dev_learning_curve'], 'dev')
