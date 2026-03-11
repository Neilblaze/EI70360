def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    tp = fp = tn = fn = 0

    for y_true, y_pred in zip(actual, predicted):
        if y_true == 1 and y_pred == 1:
            tp += 1  # TP
        elif y_true == 0 and y_pred == 1:
            fp += 1  # FP
        elif y_true == 1 and y_pred == 0:
            fn += 1  # FN
        elif y_true == 0 and y_pred == 0:
            tn += 1  # TN

    confusion_matrix = [[tp, fn], [fp, tn]]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    specificity = tn / (tn + fp)
    negativePredictive = tn / (tn + fn)

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negativePredictive, 3),
    )


actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))
