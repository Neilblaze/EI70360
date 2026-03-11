import numpy as np


def jaccard_index(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    intersection = TP
    union = TP + FP + FN
    result = intersection / union if union != 0 else 0.0
    return round(result, 3)


y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
print(jaccard_index(y_true, y_pred))
