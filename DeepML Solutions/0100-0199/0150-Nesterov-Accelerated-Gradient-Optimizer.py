import numpy as np


def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    lookahead_param = parameter - momentum * velocity
    grad = grad_fn(lookahead_param)
    new_velocity = momentum * velocity + learning_rate * grad
    new_param = parameter - new_velocity
    return np.round(new_param, 5), np.round(new_velocity, 5)
