import numpy as np


def pegasos_kernel_svm(
    data, labels, kernel="linear", lambda_val=0.01, iterations=100, sigma=1.0
):
    n_samples, n_features = data.shape
    alphas = np.zeros(n_samples)
    b = 0.0

    if kernel == "linear":
        kernel_func = lambda x1, x2: np.dot(x1, x2)
    elif kernel == "rbf":
        kernel_func = lambda x1, x2: np.exp(
            (-(np.linalg.norm(x1 - x2) ** 2)) / (2 * (sigma**2))
        )

    for iteration in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * iteration)

            margin = np.zeros((n_samples,))
            for j in range(n_samples):
                margin[j] = alphas[j] * labels[j] * kernel_func(data[j], data[i])

            decision = np.sum(margin) + b

            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return alphas, b


print(
    pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        kernel="linear",
        lambda_val=0.01,
        iterations=100,
    )
)
print(
    pegasos_kernel_svm(
        np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        np.array([1, 1, -1, -1]),
        kernel="rbf",
        lambda_val=0.01,
        iterations=100,
        sigma=0.5,
    )
)
