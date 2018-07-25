import torch
from torch.utils.data import DataLoader
from deepc.analysis import analysis
import itertools
import os.path


class Train:

    def __init__(self, model, loss_func, dataset, num_workers=0, learning_rate=1e-4,
                 optimizer=None, save_path=None, stats_path=None):
        """
        TODO: document
        :param model:
        :param loss_func:
        :param dataset:
        :param num_workers:
        :param learning_rate:
        :param optimizer:
        :param save_path:
        """
        self._model = model
        self._loss_func = loss_func
        self._dataset = dataset
        self._data_loader = DataLoader(self._dataset, shuffle=True, num_workers=num_workers)

        if optimizer:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._save_path = save_path
        self._stats_path = stats_path

    def run(self, max_epochs=float('inf')):
        """
        TODO: document
        :return:
        """
        stats = analysis.load(self._stats_path) if os.path.isfile(self._stats_path) else analysis.Analysis("Train")

        for epoch in itertools.count():
            if epoch > max_epochs:
                break

            for sample in self._data_loader:

                local_data, local_labels = sample['image'], sample['labels']

                pred = self._model(local_data.permute([0, 3, 1, 2]).float())

                loss = self._loss_func(pred.squeeze(0), local_labels.squeeze(0))
                print(f"epoch:{epoch}, loss:{loss}")
                stats.loss.append(loss)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

            stats.epoch()

            if self._save_path:
                torch.save(self._model.state_dict(), self._save_path)

            if self._stats_path:
                analysis.save(stats, self._stats_path)
