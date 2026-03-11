def confusion_matrix(data):
    matrix = [[0, 0], [0, 0]]

    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            matrix[0][0] += 1  # TP
        elif y_true == 0 and y_pred == 1:
            matrix[1][0] += 1  # FP
        elif y_true == 1 and y_pred == 0:
            matrix[0][1] += 1  # FN
        elif y_true == 0 and y_pred == 0:
            matrix[1][1] += 1  # TN

    return matrix


data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
cm = confusion_matrix(data)
print(cm)
