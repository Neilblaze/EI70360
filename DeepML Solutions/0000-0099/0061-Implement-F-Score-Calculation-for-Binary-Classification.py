import numpy as np


def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))

    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0

    if tp + fn > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0

    if ((beta**2) * prec + rec) > 0:
        f_beta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    else:
        f_beta = 0

    return round(f_beta, 3)


y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
beta = 1

print(f_score(y_true, y_pred, beta))
