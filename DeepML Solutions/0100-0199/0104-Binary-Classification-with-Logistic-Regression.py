import numpy as np


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = np.dot(X, weights) + bias
    probabilities = 1 / (1 + np.exp(-z))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions