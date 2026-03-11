import numpy as np

def ucb_action(counts, values, t, c):
    ucb_values = values + c * np.sqrt(np.log(t) / (counts + 1e-8))
    return np.argmax(ucb_values)
