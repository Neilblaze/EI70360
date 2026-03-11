import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    combinations = []
    for i in range(0, degree + 1):
        combinations.append(combinations_with_replacement(range(n_features), i))

    flat_combinations = [item for combination in combinations for item in combination]

    new_features = np.empty((n_samples, len(flat_combinations)))
    for i, combination in enumerate(flat_combinations):
        new_features[:, i] = np.prod(X[:, combination], axis=1)

    return new_features


X = np.array([[2, 3], [3, 4], [5, 6]])
degree = 2
output = polynomial_features(X, degree)
print(output)
