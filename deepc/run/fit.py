import torch
import numpy as np
import time
from deepc.run.checkpoints import save_checkpoints, update_checkpoints


def fit(model, criterion, optimizer, data_loader, n_epochs, device=torch.device('cuda'), logger=None, start_epoch=0,
        checkpoints=None, checkpoints_file_path=None, save_freq=None):
    last_save_time = time.time()
    learning_curve = []
    for epoch_idx in range(n_epochs):
        epoch_losses = []
        for sample in data_loader:
            # TODO: rename 'image' and 'labels' to 'x' and 'y'
            x_batch, y_batch = sample['image'].to(device), sample['labels'].to(device)
            # TODO: permute in dataset
            pred_batch = model(x_batch.permute([0, 3, 1, 2]))
            loss = criterion(pred_batch, y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if (time.time() - last_save_time) / 60.0 >= save_freq:
                checkpoints = update_checkpoints(checkpoints, model=model, optimizer=optimizer,
                                                 train_learning_curve=learning_curve)
                save_checkpoints(checkpoints, checkpoints_file_path)
                last_save_time = time.time()
        learning_curve.append(np.mean(epoch_losses))
        if logger:
            logger.info(f"Train epoch - epoch:{start_epoch + 1 + epoch_idx} avg-loss:{np.mean(epoch_losses)}")

    return learning_curve
