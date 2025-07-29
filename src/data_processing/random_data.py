from math import prod

import numpy as np


def random_walk(size):
    return np.clip(1 + np.cumsum(np.random.uniform(low=-1., high=1, size=size), axis=1), 0.1, 2.)


def random_cycle_data(size: tuple, offset=2, denom=5.):
    data = np.arange(prod(size)).reshape(size)
    return (np.sin(data) + offset) / denom + 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = random_cycle_data((2, 35))
    plt.plot(data.T)
    plt.show()
