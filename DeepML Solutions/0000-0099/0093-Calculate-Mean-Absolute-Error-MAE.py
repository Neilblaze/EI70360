import numpy as np


def mae(y_true, y_pred):
    val = np.mean(np.abs(y_true - y_pred))
    return round(val, 3)


y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
result = mae(y_true, y_pred)
print(result)
