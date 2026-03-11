import numpy as np


def translate_object(points, tx, ty):
    result = []
    for row in points:
        result.append([row[0] + tx, row[1] + ty])

    return result


points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))

# Expected Output:
# [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
