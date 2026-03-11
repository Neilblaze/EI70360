import numpy as np


def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)
    rows, cols = X.shape

    if replacements:
        len_subset = rows
    else:
        len_subset = rows // 2

    result = []
    for i in range(n_subsets):
        idx = np.random.choice(rows, len_subset, replace=replacements)
        result.append((X[idx].tolist(), y[idx].tolist()))

    return result


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

y = np.array([1, 2, 3, 4, 5])
n_subsets = 3
replacements = False

result = get_random_subsets(X, y, n_subsets, replacements)
print(result)
