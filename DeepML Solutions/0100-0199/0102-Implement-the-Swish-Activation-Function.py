import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def swish(x: float) -> float:
    return x * sigmoid(x)


print(swish(1))
