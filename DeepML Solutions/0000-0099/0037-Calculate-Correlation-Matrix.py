import numpy as np


def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    covariance_matrix = (X_centered.T @ Y_centered) / (X_centered.shape[0] - 1)

    stddev_X = np.std(X, axis=0, ddof=1)
    stddev_Y = np.std(Y, axis=0, ddof=1)

    correlation_matrix = covariance_matrix / np.outer(stddev_X, stddev_Y)
    return correlation_matrix


X = np.array([[1, 2], [3, 4], [5, 6]])
output = calculate_correlation_matrix(X)
print(output)
