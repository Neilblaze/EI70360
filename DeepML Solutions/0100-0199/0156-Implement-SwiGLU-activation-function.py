import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    d = x.shape[1] // 2
    x1 = x[:, :d]
    x2 = x[:, d:]
    swish = x2 * sigmoid(x2)
    result = x1 * swish
    return np.round(result, 4)
