import torch
from torch.utils.data import DataLoader
import itertools


class Train:

    def __init__(self, model, loss_func, dataset, num_workers=4, max_epochs=float('inf'), learning_rate=1e-4,
                 optimizer=None, save_path=None):
        """
        TODO: document
        :param model:
        :param loss_func:
        :param dataset:
        :param num_workers:
        :param max_epochs:
        :param learning_rate:
        :param optimizer:
        :param save_path:
        """
        self._model = model
        self._loss_func = loss_func
        self._dataset = dataset
        self._data_loader = DataLoader(self._dataset, shuffle=True, num_workers=num_workers)
        self._max_epochs = max_epochs

        if optimizer:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._save_path = save_path

    def run(self):
        """
        TODO: document
        :return:
        """
        for epoch in itertools.count():
            if epoch > self._max_epochs:
                break

            for sample in self._data_loader:

                local_data, local_labels = sample['image'], sample['labels']

                pred = self._model(local_data.permute([2, 0, 1]).unsqueeze(0).float()).squeeze(0)

                loss = self._loss_func(pred, local_labels)
                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

            if self._save_path:
                torch.save(self._model.state_dict(), self._save_path)
