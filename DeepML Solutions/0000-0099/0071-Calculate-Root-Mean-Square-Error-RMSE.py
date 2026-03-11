import numpy as np


def rmse(y_true, y_pred):
    return round(np.sqrt(np.mean((y_pred - y_true) ** 2)), 3)


y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
print(rmse(y_true, y_pred))
