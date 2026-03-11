import numpy as np


def gradient_descent(
    X, y, weights, learning_rate, n_iterations, batch_size=1, method="batch"
):
    for _ in range(n_iterations):
        if method == "batch":
            gradients = (2 * X.T.dot(X.dot(weights) - y)) / len(y)
            weights -= learning_rate * gradients

        elif method == "stochastic":
            for i in range(len(y)):
                gradients = 2 * X[i].T.dot(X[i].dot(weights) - y[i])
                weights -= learning_rate * gradients

        elif method == "mini_batch":
            for i in range(0, len(y), batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                gradients = (
                    2 / batch_size * X_batch.T.dot(X_batch.dot(weights) - y_batch)
                )
                weights -= learning_rate * gradients

    return weights


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_iterations = 100

# Test Batch Gradient Descent
final_weights = gradient_descent(
    X, y, weights, learning_rate, n_iterations, method="batch"
)
print(final_weights)

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_iterations = 100

# Test Stochastic Gradient Descent
final_weights = gradient_descent(
    X, y, weights, learning_rate, n_iterations, method="stochastic"
)
print(final_weights)

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_iterations = 100
batch_size = 2

# Test Mini-Batch Gradient Descent
final_weights = gradient_descent(
    X, y, weights, learning_rate, n_iterations, batch_size, method="mini_batch"
)
print(final_weights)
