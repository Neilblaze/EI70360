import numpy as np


def precision(y_true, y_pred):
    TP = 0
    FP = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1

        if y_pred[i] == 1 and y_pred[i] != y_true[i]:
            FP += 1

    return TP / (TP + FP)


y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)
