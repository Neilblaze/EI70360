import math


def poisson_probability(k, lam):
    v1 = (math.e**-lam) * (lam**k)
    v2 = math.perm(k)
    val = v1 / v2
    return round(val, 5)
