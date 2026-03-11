import numpy as np


def to_categorical(x, n_col=None):
    arr = []
    for val in x:
        data = [0] * (max(x) + 1)
        data[val] = 1
        arr.append(data)

    return arr


x = np.array([0, 1, 2, 1, 0])
ohe = to_categorical(x, n_col=None)
print(ohe)
