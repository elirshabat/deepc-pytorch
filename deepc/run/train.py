import torch
from torch.utils.data import DataLoader
from deepc.analysis import analysis
import itertools
import os.path
import logging


class Train:

    def __init__(self, loader, model, criterion, optimizer, cpu=False,
                 start_epoch=0, start_iteration=0, iteration_size=None):
        self._model = model
        self._criterion = criterion
        self._cuda_available = torch.cuda.device_count() > 0
        self._loader = loader
        self._data_enumerator = enumerate(self._loader)
        self._optimizer = optimizer
        self._logger = logging.getLogger('train')
        self._cpu = cpu
        self.epoch = start_epoch
        self.iteration = start_iteration
        self.iteration_size = iteration_size

    def run(self):
        total_loss = 0.0
        epoch_done = False
        if self.iteration_size is not None:
            local_iteration_size = self.iteration_size
        else:
            local_iteration_size = len(self._loader)//self._loader.batch_size

        for t in range(local_iteration_size):
            try:
                batch_index, sample = next(self._data_enumerator)

                if not self._cpu:
                    data_batch, labels_batch = sample['image'].cuda(), sample['labels'].cuda()
                else:
                    data_batch, labels_batch = sample['image'], sample['labels']

                pred_batch = self._model(data_batch.permute([0, 3, 1, 2]))
                loss = self._criterion(pred_batch, labels_batch)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()

                self._logger.debug(
                    f"Train step - epoch:{self.epoch} iteration:{self.iteration} batch:{batch_index} "
                    f"t:{t}/{local_iteration_size} loss:{loss.item()}")

            except StopIteration:
                self.epoch += 1
                epoch_done = True
                del self._data_enumerator
                self._data_enumerator = enumerate(self._loader)

        self.iteration += 1

        return total_loss/local_iteration_size, epoch_done

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
