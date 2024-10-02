# gets and npz matrix for sys argv
# plots the state matrix with imshow

import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_state(state):
    plt.imshow(state, cmap="viridis")
    plt.show()


if __name__ == "__main__":
    state = np.load(sys.argv[1])["state"]
    plot_state(state)
