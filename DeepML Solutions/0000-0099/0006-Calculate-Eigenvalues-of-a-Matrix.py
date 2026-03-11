import numpy as np


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    return np.linalg.eigvals(matrix)


matrix = [[2, 1], [1, 2]]
eigenvalues = calculate_eigenvalues(matrix)
print(eigenvalues)
