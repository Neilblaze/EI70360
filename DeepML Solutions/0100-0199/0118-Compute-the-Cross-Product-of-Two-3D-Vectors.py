import numpy as np

def cross_product(a, b):
    #          | a2b3 - a3b2 |
    # a x b =  | a3b1 - a1b3 |
    #          | a1b2 - a2b1 |
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])

print(cross_product([1, 0, 0], [0, 1, 0]))