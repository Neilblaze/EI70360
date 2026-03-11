import numpy as np


def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    velocity = momentum * velocity + learning_rate * grad
    parameter = parameter - velocity
    return np.round(parameter, 5), np.round(velocity, 5)