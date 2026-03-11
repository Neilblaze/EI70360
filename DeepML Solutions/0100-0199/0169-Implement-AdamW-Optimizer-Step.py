import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * w)
    return np.round(w, 5), np.round(m, 5), np.round(v, 5)
