import numpy as np


def softsign(x: float) -> float:
    val = x / (1 + np.abs(x))
    return round(val, 4)


print(softsign(1))
