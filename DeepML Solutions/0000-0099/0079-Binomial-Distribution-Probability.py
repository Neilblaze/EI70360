import math


def binomial_probability(n, k, p):
    probability = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    return round(probability, 5)
