import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)

    theta = np.linalg.solve(X_transpose_X, X_transpose_y)
    theta = (
        np.round(
            theta,
            4,
        )
        .flatten()
        .tolist()
    )
    return theta


X = [[1, 1], [1, 2], [1, 3]]
y = [1, 2, 3]
theta = linear_regression_normal_equation(X, y)
print(theta)
