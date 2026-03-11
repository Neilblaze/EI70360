import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b)
    M_inv = np.diag(1 / np.diag(A))
    N = A - np.diag(np.diag(A))

    for _ in range(n):
        x = np.dot(M_inv, b - np.dot(N, x))
    return np.round(x, 4).tolist()


A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
b = [-1, 2, 3]
n = 2
x = solve_jacobi(A, b, n)
print(x)
