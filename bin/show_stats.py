import os.path
import sys
from deepc.analysis import analysis
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="path to stats file")
    args = parser.parse_args()

    curr_dir = os.path.abspath(os.path.dirname(__file__))
    repo_dir = os.path.join(curr_dir, "..")
    sys.path.append(repo_dir)

    train_stats = analysis.load(args.path)
    train_stats.plot()

    plt.show(block=True)
