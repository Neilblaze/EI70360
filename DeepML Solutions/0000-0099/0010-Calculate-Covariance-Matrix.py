import numpy as np


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(np.stack(vectors, axis=0))


vectors = [[1, 2, 3], [4, 5, 6]]
covariance_matrix = calculate_covariance_matrix(vectors)
print(covariance_matrix)
