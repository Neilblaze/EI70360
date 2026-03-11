import numpy as np


def calculate_dot_product(vec1, vec2) -> float:
    return sum(a * b for a, b in zip(vec1, vec2))
