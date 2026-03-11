import numpy as np


def rref(matrix):
    A = matrix.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[i] += A[k]
                    break

        if A[i, i] == 0:
            continue

        A[i] = A[i] / A[i, i]
        for j in range(n):
            if i != j:
                A[j] -= A[j, i] * A[i]

    return A.tolist()


matrix = np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]])

rref_matrix = rref(matrix)
print(rref_matrix)
