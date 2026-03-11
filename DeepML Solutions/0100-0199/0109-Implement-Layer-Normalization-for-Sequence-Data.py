import numpy as np


def layer_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    result = np.zeros(shape=X.shape)
    for i in range(len(X)):
        mean = np.mean(X[i], axis=-1, keepdims=True)
        var = np.var(X[i], axis=-1, keepdims=True)
        layer_norm = (X[i] - mean) / ((var + epsilon) ** 0.5) * gamma + beta
        result[i] = layer_norm

    return result


np.random.seed(42)
X = np.random.randn(2, 2, 3)
gamma = np.ones(3).reshape(1, 1, -1)
beta = np.zeros(3).reshape(1, 1, -1)
result = layer_normalization(X, gamma, beta)
print(result)
