import numpy as np


def initialisation(m, n):
    data_0 = np.random.randn(m, n)
    data_1 = np.ones((m, 1))
    return np.hstack((data_0, data_1))
