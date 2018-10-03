import torch
import numpy as np
import time
from deepc.run.checkpoints import save_checkpoints, update_checkpoints


def fit(model, criterion, optimizer, data_loader, n_epochs, device=torch.device('cuda'), logger=None, start_epoch=0,
        checkpoints=None, checkpoints_file_path=None, save_freq=None, log_freq=None):
    last_save_time = last_log_time = time.time()
    t = 0
    avg_sampling_time = avg_pred_time = avg_loss_time = avg_backward_time = 0.0
    learning_curve = []
    for epoch_idx in range(n_epochs):
        epoch_losses = []
        start_time = time.time()
        for sample in data_loader:
            t += 1
            # TODO: rename 'image' and 'labels' to 'x' and 'y'
            x_batch, y_batch = sample['image'].to(device), sample['labels'].to(device)
            avg_sampling_time += time.time() - start_time
            # TODO: permute in dataset
            start_time = time.time()
            pred_batch = model(x_batch.permute([0, 3, 1, 2]))
            avg_pred_time += time.time() - start_time
            start_time = time.time()
            loss = criterion(pred_batch, y_batch)
            avg_loss_time += time.time() - start_time
            model.zero_grad()
            start_time = time.time()
            loss.backward()
            avg_backward_time += time.time() - start_time
            optimizer.step()
            epoch_losses.append(loss.item())
            if (time.time() - last_save_time) / 60.0 >= save_freq:
                checkpoints = update_checkpoints(checkpoints, model=model, optimizer=optimizer,
                                                 train_learning_curve=learning_curve)
                save_checkpoints(checkpoints, checkpoints_file_path)
                last_save_time = time.time()
            if (time.time() - last_log_time) / 60.0 >= log_freq:
                logger.debug(f"Learning stats - epoch:{start_epoch + 1 + epoch_idx} avg-loss:{np.mean(epoch_losses)}")
                logger.debug(f"Time stats - avg_sampling_time:{avg_sampling_time/t} avg_pred_time:{avg_pred_time/t} "
                             f"avg_loss_time:{avg_loss_time/t} avg_backward_time:{avg_backward_time/t}")
                last_log_time = time.time()
        learning_curve.append(np.mean(epoch_losses))
        if logger:
            logger.info(f"Train epoch - epoch:{start_epoch + 1 + epoch_idx} avg-loss:{np.mean(epoch_losses)}")

    return learning_curve
