import numpy as np
from collections import Counter


def gini_impurity(y):
    gini = 0
    items = Counter(y)

    for item in items.keys():
        gini += (items[item] / len(y)) ** 2

    return round(gini, 3)


y = [0, 1, 1, 1, 0]
print(gini_impurity(y))
