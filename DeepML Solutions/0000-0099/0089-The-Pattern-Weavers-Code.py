import numpy as np


def softmax(values):
    e_x = np.exp(values - np.max(values))
    return e_x / np.sum(e_x, axis=0)


def pattern_weaver(n, crystal_values, dimension):
    attention_scores = np.outer(crystal_values, crystal_values) / np.sqrt(dimension)
    softmax_scores = np.apply_along_axis(softmax, axis=1, arr=attention_scores)
    result = np.dot(softmax_scores, crystal_values)
    return np.round(result, 4).tolist()


n = 5
crystal_values = np.array([4, 2, 7, 1, 9])
dimension = 1
result = pattern_weaver(n, crystal_values, dimension)
print(result)
