import pickle
import matplotlib.pyplot as plt


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save(analysis, file):
    with open(file, 'wb') as f:
        pickle.dump(analysis, f)


class Analysis:

    def __init__(self, name):
        self.name = name
        self.loss_curve = []
        self.loss = []

    def epoch(self):
        if self.loss:
            self.loss_curve.append(sum(self.loss)/len(self.loss))
        self.loss = []

    def plot(self):
        fig, loss_ax = plt.subplots(1, 1)
        loss_ax.plot(self.loss_curve)
        fig.suptitle(f"{self.name} - stats")
        loss_ax.set_title("Loss Curve")
        loss_ax.set_xlabel("epoch")
        loss_ax.set_ylabel("loss")
