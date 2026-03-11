import numpy as np


def softplus(x: float) -> float:
    return round(np.log(1 + np.exp(x)), 4)


print(softplus(2))
