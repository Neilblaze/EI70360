import numpy as np


def pca(data: np.ndarray, k: int) -> list[list[int | float]]:
    mean = sum(data) / len(data)
    std = (sum((i - mean) ** 2 for i in data) / len(data)) ** 0.5
    X_std = (data - mean) / std
    cov_mat = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    idx = eigen_vals.argsort()[::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]

    principal_components = eigen_vecs[:, :k].tolist()
    return np.round(principal_components, 4)


data = np.array([[1, 2], [3, 4], [5, 6]])
k = 1
principal_components = pca(data, k)
print(principal_components)
