import numpy as np
import matplotlib.pyplot as plt

dataset = {f"experience{i}": np.random.randn(100, 3) for i in range(4)}


def graphique(dataset):
    fig, ax = plt.subplots(len(dataset))
    for i in range(len(ax)):
        key = f"experience{i}"
        ax[i].plot(dataset[key])
        ax[i].set_title(key)
    plt.show()


graphique(dataset)
