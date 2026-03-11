import math


def elu(x: float, alpha: float = 1.0) -> float:
    val = alpha * (math.exp(x) - 1) if x < 0 else float(x)
    return round(val, 4)


print(elu(0))
