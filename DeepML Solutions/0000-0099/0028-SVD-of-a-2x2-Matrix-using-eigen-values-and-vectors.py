import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    ATA = A.T @ A
    eigvals_V, V = np.linalg.eig(ATA)
    s = np.sqrt(eigvals_V)
    sort_indices = np.argsort(-s)
    s = s[sort_indices]
    V = V[:, sort_indices]
    U = A @ V / s
    return U, s, V.T


A = [[-10, 8], [10, -1]]

U, Sigma, Vt = svd_2x2(A)
print(U)
print(Sigma)
print(Vt)
