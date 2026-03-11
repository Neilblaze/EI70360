import numpy as np

def f(x, a, b, c, d):
    return c + (d - c) / (b - a) * (x - a)

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    a = np.min(values)
    b = np.max(values)

    result = []
    for value in values:
        result.append(f(value, a, b, c, d))

    return result