import numpy as np


def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2


y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
print(r_squared(y_true, y_pred))
