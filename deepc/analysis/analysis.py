import pickle
import matplotlib.pyplot as plt
import numpy as np


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save(analysis, file):
    with open(file, 'wb') as f:
        pickle.dump(analysis, f)


class Analysis:

    def __init__(self, name, iteration_size=1e9):
        self._name = name
        self._iteration_size = iteration_size
        self._epoch_loss_curve = []
        self._epoch_loss = []
        self._iteration_loss = []
        self._iteration_loss_curve = []

    def _end_epoch(self):
        if self._epoch_loss:
            self._epoch_loss_curve.append(sum(self._epoch_loss) / len(self._epoch_loss))
        self._epoch_loss = []
        self.plot()

    def _end_iteration(self):
        if self._iteration_loss:
            self._iteration_loss_curve.append(sum(self._iteration_loss) / len(self._iteration_loss))
        self._iteration_loss = []
        self.plot()

    def step(self, loss=None, epoch_end=False):
        if loss is not None:
            self._epoch_loss.append(loss)
            self._iteration_loss.append(loss)

        if len(self._iteration_loss) >= self._iteration_size:
            self._end_iteration()

        if epoch_end:
            self._end_epoch()

    def plot(self):
        grid_rows, grid_cols = 1, 2

        if not plt.fignum_exists('stats'):
            fig, axes = plt.subplots(grid_rows, grid_cols, sharex=True, squeeze=False, num='stats')
            fig.suptitle(f"{self._name} - stats")

            axes[0][0].set_title("epoch vs. loss curve")
            axes[0][0].set_xlabel("epoch")
            axes[0][0].set_ylabel("loss")

            axes[0][1].set_title("iteration vs. loss curve")
            axes[0][1].set_xlabel(f"iteration (size={self._iteration_size})")
            axes[0][1].set_ylabel("loss")

        fig = plt.figure('stats')

        for ax in fig.get_axes():
            for artist in ax.lines + ax.collections:
                artist.remove()

        axes = np.array(fig.get_axes()).reshape(grid_rows, grid_cols)
        axes[0][0].plot(self._epoch_loss_curve)
        axes[0][1].plot(self._iteration_loss_curve)

        plt.pause(0.01)
