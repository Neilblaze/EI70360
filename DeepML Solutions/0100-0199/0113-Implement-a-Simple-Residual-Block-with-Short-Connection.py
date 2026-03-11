import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    identity = x.copy()
    x = np.dot(w1, x)
    x = np.maximum(0, x)
    x = np.dot(w2, x)
    x += identity
    x = np.maximum(0, x)
    return x