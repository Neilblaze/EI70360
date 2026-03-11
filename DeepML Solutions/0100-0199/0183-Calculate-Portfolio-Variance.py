import numpy as np


def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    n = len(weights)
    result = 0
    for i in range(n):
        for j in range(n):
            v = cov_matrix[i][j] * weights[i] * weights[j]
            result += v

    return result

cov_matrix = [[0.1, 0.02], [0.02, 0.15]]
weights = [0.6, 0.4]
result = calculate_portfolio_variance(cov_matrix, weights)
print(result)
