import numpy as np


def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    G += grad ** 2
    parameter -= learning_rate / (np.sqrt(G) + epsilon) * grad
    return np.round(parameter, 5), np.round(G, 5)