import numpy as np


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    reshaped_matrix = np.array(a).reshape(new_shape).tolist()
    return reshaped_matrix


a = [[1, 2, 3, 4], [5, 6, 7, 8]]
new_shape = (4, 2)
reshaped_matrix = reshape_matrix(a, new_shape)
print(reshaped_matrix)
