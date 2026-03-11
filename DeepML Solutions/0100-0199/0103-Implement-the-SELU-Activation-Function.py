import numpy as np


def selu(x: float) -> float:
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


print(selu(-1.0))
