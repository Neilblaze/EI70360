import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.array(A)
    AtA = np.dot(A.T, A)
    eigen_vals, eigen_vecs = np.linalg.eigh(AtA)
    ncols = np.argsort(eigen_vals)[::-1]
    singular_values = np.sqrt(eigen_vals[ncols])
    eigen_vecs = eigen_vecs[:, ncols]
    return eigen_vecs, singular_values, eigen_vecs.T


A = [[2, 1], [1, 2]]

SVD = svd_2x2_singular_values(A)
print(SVD)
