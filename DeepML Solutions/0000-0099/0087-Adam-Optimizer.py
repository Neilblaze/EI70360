import numpy as np


def adam_optimizer(
    parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - (beta1**t))
    v_hat = v / (1 - (beta2**t))
    parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return np.round(parameter, 5), np.round(m, 5), np.round(v, 5)


parameter, m, v = adam_optimizer(1.0, 0.1, 0.0, 0.0, 1)
print(parameter)
print(m)
print(v)
