import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    e_scores = np.exp(scores - np.max(scores))
    return np.log(e_scores / e_scores.sum())


A = np.array([1, 2, 3])
print(log_softmax(A))
