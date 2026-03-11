import numpy as np


def l1_regularization_gradient_descent(
    X: np.array,
    y: np.array,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, weights) + bias

        dW = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + alpha * np.sign(weights)
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= learning_rate * dW
        bias -= learning_rate * db

    return weights, bias


X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])

alpha = 0.1
weights, bias = l1_regularization_gradient_descent(
    X, y, alpha=alpha, learning_rate=0.01, max_iter=1000
)

print(weights)
print(bias)
