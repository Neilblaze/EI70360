import numpy as np


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B = np.array(B)
    C = np.array(C)
    transformed = np.linalg.solve(C.T, B).T
    P = []
    for row in transformed:
        row = [round(item, 4) for item in row]
        P.append(row)
    return P


B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
C = [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]
P = transform_basis(B, C)
print(P)
