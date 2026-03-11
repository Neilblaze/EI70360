def calculate_f1_score(y_true, y_pred):
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for y_t, y_p in zip(y_true, y_pred):
        if y_t == 1 and y_p == 1:
            tp += 1
        elif y_t != 1 and y_p == 1:
            fp += 1
        elif y_t == 1 and y_p != 1:
            fn += 1

    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0.0

    if tp + fn > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0.0

    beta = 1
    if ((beta**2) * prec + rec) > 0:
        f1 = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    else:
        f1 = 0.0

    return round(f1, 3)


y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]
result = calculate_f1_score(y_true, y_pred)
print(result)
