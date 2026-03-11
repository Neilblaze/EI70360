import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n):
        max_row = np.argmax(np.abs(A[k:, k])) + k
        if A[max_row, k] == 0:
            raise ValueError("Matrix is singular.")
        
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = float((b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i])

    return x.tolist()
