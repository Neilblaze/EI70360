import numpy as np


def divide_on_feature(X, feature_i, threshold):
    upper_arr = []
    lower_arr = []

    for row in X:
        if row[feature_i] < threshold:
            lower_arr.append(row)
        else:
            upper_arr.append(row)

    return [np.array(upper_arr), np.array(lower_arr)]


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

feature_i = 0
threshold = 5

result = divide_on_feature(X, feature_i, threshold)
print(result)
