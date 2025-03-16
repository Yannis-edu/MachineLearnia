import numpy as np

np.random.seed(0)
A = np.random.randint(0, 100, [10, 5])

A = (A - A.mean(axis=0)) / A.std(axis=0)

print(A)
