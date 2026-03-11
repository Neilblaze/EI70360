import numpy as np


def make_diagonal(x):
    matrix = [[0 for _ in range(len(x))] for _ in range(len(x))]
    for i in range(len(x)):
        matrix[i][i] = x[i]
    return matrix


x = np.array([1, 2, 3])
output = make_diagonal(x)
print(output)
