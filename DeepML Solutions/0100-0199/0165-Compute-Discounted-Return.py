import numpy as np

def discounted_return(rewards, gamma):
    n = len(rewards)
    discounts = gamma ** np.arange(n)
    return np.sum(rewards * discounts)
