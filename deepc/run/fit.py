import torch
import numpy as np
import time
import itertools
from deepc.run.checkpoints import save_checkpoints, update_checkpoints


def fit(model, criterion, optimizer, data_loader, n_epochs, device=torch.device('cuda'), logger=None, start_epoch=0,
        checkpoints=None, checkpoints_file_path=None, save_freq=None, log_freq=None):
    # Initialize parameters
    last_save_time = last_log_time = time.time()
    t = 0
    avg_sampling_time = avg_pred_time = avg_loss_time = avg_backward_time = 0.0
    learning_curve = []

    # Train for the given number of epoch (which can be infinite)
    epochs_range = itertools.count() if n_epochs == float('inf') else range(n_epochs)
    for epoch_idx in epochs_range:
        # Reset epoch parameters
        epoch_losses = []
        epoch_start_time = time.time()
        # Train using samples from the dataset
        for sample in data_loader:
            t += 1
            # Sample
            # TODO: rename 'image' and 'labels' to 'x' and 'y'
            x_batch, y_batch = sample['image'].to(device), sample['labels'].to(device)
            avg_sampling_time += time.time() - epoch_start_time
            # Forward pass
            # TODO: permute in dataset
            pred_start_time = time.time()
            pred_batch = model(x_batch.permute([0, 3, 1, 2]))
            avg_pred_time += time.time() - pred_start_time
            # Compute loss
            loss_start_time = time.time()
            loss = criterion(pred_batch, y_batch)
            avg_loss_time += time.time() - loss_start_time
            # Update model's parameters
            model.zero_grad()
            backward_start_time = time.time()
            loss.backward()
            avg_backward_time += time.time() - backward_start_time
            optimizer.step()
            # Update epoch stats
            epoch_losses.append(loss.item())
            # Save checkpoints if enough time has passed
            if (time.time() - last_save_time) / 60.0 >= save_freq:
                checkpoints = update_checkpoints(checkpoints, model=model, train_learning_curve=learning_curve)
                save_checkpoints(checkpoints, checkpoints_file_path)
                last_save_time = time.time()
            # Log messages on some iterations
            if (time.time() - last_log_time) / 60.0 >= log_freq:
                logger.info(f"Learning stats - epoch:{start_epoch + 1 + epoch_idx} avg-loss:{np.mean(epoch_losses)}")
                logger.debug(f"Time stats - avg_sampling_time:{avg_sampling_time/t} avg_pred_time:{avg_pred_time/t} "
                             f"avg_loss_time:{avg_loss_time/t} avg_backward_time:{avg_backward_time/t}")
                last_log_time = time.time()
        # Update learning stats
        learning_curve.append(np.mean(epoch_losses))
        # Log epoch stats
        logger.info(f"Train epoch - epoch:{start_epoch + 1 + epoch_idx} avg-loss:{np.mean(epoch_losses)}")

    return learning_curve
