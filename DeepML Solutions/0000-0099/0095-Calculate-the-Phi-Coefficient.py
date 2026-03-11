def phi_corr(x: list[int], y: list[int]) -> float:
    tp = fp = tn = fn = 0

    for y_true, y_pred in zip(x, y):
        if y_true == 1 and y_pred == 1:
            tp += 1  # TP
        elif y_true == 0 and y_pred == 1:
            fp += 1  # FP
        elif y_true == 1 and y_pred == 0:
            fn += 1  # FN
        elif y_true == 0 and y_pred == 0:
            tn += 1  # TN

    n = tn + tp + fn + fp
    s = (tp + fn) / n
    p = (tp + fp) / n

    val = ((tp * tn) - (fp * fn)) / (
        ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    )
    return round(val, 4)


print(phi_corr([1, 1, 0, 0], [0, 0, 1, 1]))
