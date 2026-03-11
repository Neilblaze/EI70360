import math


def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return result


z = 0
result = sigmoid(z)
print(result)
