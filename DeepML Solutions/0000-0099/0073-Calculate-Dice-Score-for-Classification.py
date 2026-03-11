import numpy as np


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice_score = (
        (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
        if intersection != 0
        else 0.0
    )
    return round(dice_score, 3)


y_true = np.array([1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 0, 0, 1])
print(dice_score(y_true, y_pred))
