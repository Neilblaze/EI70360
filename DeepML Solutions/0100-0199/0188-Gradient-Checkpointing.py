import numpy as np


def checkpoint_forward(funcs, input_arr):
    arr = input_arr.copy()
    for f in funcs:
        arr = f(arr)

    return arr
