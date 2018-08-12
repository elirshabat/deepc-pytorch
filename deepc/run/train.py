import torch
from torch.utils.data import DataLoader
from deepc.analysis import analysis
import itertools
import os.path
import logging


class Train:

    def __init__(self, loader, model, criterion, optimizer, cpu=False, log_freq=1):
        self._model = model
        # TODO: rename
        self._criterion = criterion
        self._cuda_available = torch.cuda.device_count() > 0
        self._loader = loader
        self._optimizer = optimizer
        self._logger = logging.getLogger('train')
        self._cpu = cpu
        self._log_freq = log_freq

    def run(self, epoch, iteration, iteration_size):
        t_train, iteration_loss = 0, 0.0
        while t_train < iteration_size:
            try:
                sample = next(self._loader)
                t_train += 1

                if not self._cpu:
                    data_batch, labels_batch = sample['image'].cuda(), sample['labels'].cuda()
                else:
                    data_batch, labels_batch = sample['image'], sample['labels']

                pred_batch = self._model(data_batch.permute([0, 3, 1, 2]))
                loss = self._criterion(pred_batch, labels_batch)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

                t_train += 1
                iteration_loss += loss.item()

                if t_train % self._log_freq == 0:
                    self._logger.info(
                        f"Train step - epoch:{epoch} iteration:{iteration} mini-batch:{t_train} loss:{loss.item()}")

            except StopIteration:
                epoch += 1

        return epoch, iteration_loss/iteration_size

        # TODO: implement in Validation
        # if self._dev_set:
        #
        #     iteration_loss = 0.0
        #
        #     with torch.no_grad():
        #
        #         for sample in self._dev_set_loader:
        #
        #             if self._cuda_available:
        #                 data_batch, labels_batch = sample['image'].cuda(), sample['labels'].cuda()
        #             else:
        #                 data_batch, labels_batch = sample['image'], sample['labels']
        #
        #             pred_batch = self._model(data_batch.permute([0, 3, 1, 2]))
        #             loss = self._criterion(pred_batch, labels_batch)
        #
        #             t_dev += 1
        #             iteration_loss += loss.item()
        #
        #             self._logger.debug(f"dev step - epoch:{epoch}, loss:{loss.item()}")
        #             dev_stats.step(loss=loss.item())
        #
        #             if self._iteration_size and t_dev % self._iteration_size == 0:
        #
        #                 self._logger.info(f"dev iteration - avg_loss:{sum_loss/self._iteration_size}")
        #                 iteration_loss = 0.0
        #
        #                 if self._dev_stats_path:
        #                     analysis.save(dev_stats, self._dev_stats_path)
        #
        #                 if self._interactive:
        #                     dev_stats.plot()
        #
        #         dev_stats.step(epoch_end=True)
        #
        #         if self._dev_stats_path:
        #             analysis.save(dev_stats, self._dev_stats_path)
        #
        #         if self._interactive:
        #             dev_stats.plot()
