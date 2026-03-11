def prelu(x: float, alpha: float = 0.25) -> float:
    return x if x > 0 else alpha * x
