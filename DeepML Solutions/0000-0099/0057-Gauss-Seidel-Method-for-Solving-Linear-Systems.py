import numpy as np


def gauss_seidel(A, b, n, x_ini=None):
    if x_ini != None:
        x = x_ini
    else:
        x = np.zeros_like(b, dtype=np.float32)

    for _ in range(n):
        rows, cols = A.shape
        for i in range(rows):
            x_new = b[i]
            for j in range(cols):
                if i != j:
                    x_new = x_new - A[i][j] * x[j]

            x[i] = x_new / A[i][i]

    return list(x)


A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

n = 100
print(gauss_seidel(A, b, n))
