import math


def normal_pdf(x, mean, std_dev):
    v1 = 1 / (std_dev * math.sqrt(2 * math.pi))
    v2 = math.e ** (-0.5 * (((x - mean) / std_dev) ** 2))
    val = v1 * v2
    return round(val, 5)


print(normal_pdf(x=16, mean=15, std_dev=2.04))
