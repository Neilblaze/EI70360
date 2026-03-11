import numpy as np


def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    new_u = rho * u + (1 - rho) * grad**2
    rms_grad = np.sqrt(new_u + epsilon)
    rms_updates = np.sqrt(v + epsilon)
    delta = - (rms_updates / rms_grad) * grad
    updated_parameter = parameter + delta
    new_v = rho * v + (1 - rho) * delta**2
    return np.round(updated_parameter, 5), np.round(new_u, 5), np.round(new_v, 5) 