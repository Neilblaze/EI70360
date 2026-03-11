import numpy as np

def tanh(x: float):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    return gamma * tanh(x * alpha) + beta
