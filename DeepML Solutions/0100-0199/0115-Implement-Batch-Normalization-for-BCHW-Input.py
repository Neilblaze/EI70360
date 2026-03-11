import numpy as np
np.random.seed(42)

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    var = np.var(X, axis=(0, 2, 3), keepdims=True)
    x_norm = (X - mean) / np.sqrt(var + epsilon)
    out = gamma * x_norm + beta
    return out

B, C, H, W = 2, 2, 2, 2
X = np.random.randn(B, C, H, W)
gamma = np.ones(C).reshape(1, C, 1, 1)
beta = np.zeros(C).reshape(1, C, 1, 1)
result = batch_normalization(X, gamma, beta)
print(result)