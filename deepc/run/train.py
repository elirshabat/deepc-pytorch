import torch
from torch.utils.data import DataLoader
from deepc.analysis import analysis
import itertools
import os.path


class Train:

    def __init__(self, model, loss_func, train_set, dev_set=None, num_workers=0, learning_rate=1e-4,
                 optimizer=None, params_path=None, train_stats_path=None, dev_stats_path=None):
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
        self._train_set_loader = DataLoader(self._train_set, shuffle=True, num_workers=num_workers)
        self._dev_set_loader = DataLoader(self._dev_set, shuffle=True, num_workers=num_workers) if self._dev_set else None

        if optimizer:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._params_path = params_path
        self._train_stats_path = train_stats_path
        self._dev_stats_path = dev_stats_path

    def run(self, max_epochs=float('inf')):
        """
        TODO: document
        :return:
        """
        train_stats = analysis.load(self._train_stats_path) if os.path.isfile(
            self._train_stats_path) else analysis.Analysis("Train")

        dev_stats = analysis.load(self._dev_stats_path) if os.path.isfile(
            self._dev_stats_path) else analysis.Analysis("Dev")

        for epoch in itertools.count():
            if epoch > max_epochs:
                break

            for sample in self._train_set_loader:

                local_data, local_labels = sample['image'], sample['labels']

                pred = self._model(local_data.permute([0, 3, 1, 2]).float())

                loss = self._loss_func(pred.squeeze(0), local_labels.squeeze(0))
                print(f"epoch:{epoch}, loss:{loss}")
                train_stats.loss.append(loss)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

            train_stats.epoch()

            if self._params_path:
                torch.save(self._model.state_dict(), self._params_path)

            if self._train_stats_path:
                analysis.save(train_stats, self._train_stats_path)

            if self._dev_set:

                with torch.no_grad():

                    for sample in self._dev_set_loader:

                        local_data, local_labels = sample['image'], sample['labels']

                        pred = self._model(local_data.permute([0, 3, 1, 2]).float())

                        loss = self._loss_func(pred.squeeze(0), local_labels.squeeze(0))
                        dev_stats.loss.append(loss)

                    dev_stats.epoch()

                    if self._dev_stats_path:
                        analysis.save(dev_stats, self._dev_stats_path)
