import numpy as np


def transform_matrix(
    A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]
) -> list[list[int | float]]:
    transformed_matrix = np.linalg.inv(T).dot(A).dot(S)
    return transformed_matrix.tolist()


A = [[1, 2], [3, 4]]
T = [[2, 0], [0, 2]]
S = [[1, 1], [0, 1]]
transformed_matrix = transform_matrix(A, T, S)
print(transformed_matrix)
