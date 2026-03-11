import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(iterations):
        gradients = np.zeros(n)
        for i in range(m):
            pred = np.dot(X[i], theta)
            error = pred - y[i]
            for j in range(n):
                gradients[j] += error * X[i, j]

        gradients /= m
        theta -= alpha * gradients

    return np.round(theta, 4)


X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000

theta = linear_regression_gradient_descent(X, y, alpha, iterations)
print(theta)
