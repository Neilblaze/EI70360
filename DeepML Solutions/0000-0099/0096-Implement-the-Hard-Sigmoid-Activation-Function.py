def hard_sigmoid(x: float) -> float:
    result = 0.2 * x + 0.5
    if result > 1.0:
        return 1.0
    else:
        return result


result = hard_sigmoid(3.0)
print(result)
