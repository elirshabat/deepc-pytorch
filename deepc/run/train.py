import torch
from torch.utils.data import DataLoader
from deepc.analysis import analysis
import itertools
import os.path
import logging
from deepc.datasets.collate import collate_fn


class Train:

    def __init__(self, model, loss_func, train_set, dev_set=None, num_workers=0, learning_rate=1e-4,
                 optimizer=None, params_path=None, train_stats_path=None, dev_stats_path=None, iteration_size=None,
                 interactive=False, batch_size=1):
        """
        Tr
        :param model:
        :param loss_func:
        :param train_set:
        :param dev_set:
        :param num_workers:
        :param learning_rate:
        :param optimizer:
        :param params_path:
        :param train_stats_path:
        :param dev_stats_path:
        """
        self._model = model
        self._loss_func = loss_func
        self._train_set = train_set
        self._dev_set = dev_set
        self._train_set_loader = DataLoader(self._train_set, shuffle=True, num_workers=num_workers,
                                            batch_size=batch_size, collate_fn=collate_fn)
        self._dev_set_loader = DataLoader(self._dev_set, shuffle=True, num_workers=num_workers,
                                          batch_size=batch_size, collate_fn=collate_fn) if self._dev_set else None
        self._interactive = interactive

        if optimizer:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._params_path = params_path
        self._train_stats_path = train_stats_path
        self._dev_stats_path = dev_stats_path
        self._iteration_size = iteration_size

        self._logger = logging.getLogger('train')

    def run(self, max_epochs=float('inf')):
        """
        TODO: document
        :return:
        """
        train_stats = analysis.load(self._train_stats_path) if os.path.isfile(
            self._train_stats_path) else analysis.Analysis("Train", iteration_size=self._iteration_size)

        dev_stats = analysis.load(self._dev_stats_path) if os.path.isfile(
            self._dev_stats_path) else analysis.Analysis("Dev", iteration_size=self._iteration_size)

        t_train, t_dev = 0, 0

        for epoch in itertools.count():
            if epoch > max_epochs:
                break

            for sample in self._train_set_loader:

                local_data, local_labels, cluster_ids = sample['image'], sample['labels'], sample['cluster_ids']

                pred = self._model(local_data.permute([0, 3, 1, 2]))
                loss = self._loss_func(pred, local_labels, cluster_ids)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

                t_train += 1

                self._logger.info(f"train step - epoch:{epoch}, loss:{loss}")
                train_stats.step(loss=loss)

                if self._iteration_size and t_train % self._iteration_size == 0:

                    if self._params_path:
                        torch.save(self._model.state_dict(), self._params_path)

                    if self._train_stats_path:
                        analysis.save(train_stats, self._train_stats_path)

                    if self._interactive:
                        train_stats.plot()

            train_stats.step(epoch_end=True)

            if self._params_path:
                torch.save(self._model.state_dict(), self._params_path)

            if self._train_stats_path:
                analysis.save(train_stats, self._train_stats_path)

            if self._interactive:
                train_stats.plot()

            if self._dev_set:

                with torch.no_grad():

                    for sample in self._dev_set_loader:

                        local_data, local_labels, cluster_ids = sample['image'], sample['labels'], sample['cluster_ids']

                        pred = self._model(local_data.permute([0, 3, 1, 2]))
                        loss = self._loss_func(pred, local_labels, cluster_ids)

                        self._logger.info(f"dev step - epoch:{epoch}, loss:{loss}")
                        dev_stats.step(loss=loss)

                        if self._iteration_size and t_dev % self._iteration_size == 0:

                            if self._dev_stats_path:
                                analysis.save(dev_stats, self._dev_stats_path)

                            if self._interactive:
                                dev_stats.plot()

                    dev_stats.step(epoch_end=True)

                    if self._dev_stats_path:
                        analysis.save(dev_stats, self._dev_stats_path)

                    if self._interactive:
                        dev_stats.plot()
