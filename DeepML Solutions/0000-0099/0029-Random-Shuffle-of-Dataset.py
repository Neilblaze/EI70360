import numpy as np


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

y = np.array([1, 2, 3, 4])

X, y = shuffle_data(X, y, seed=1)
print(X)
print(y)
