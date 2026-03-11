import numpy as np


def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    a = ((mu_p - mu_q) ** 2) / (sigma_q**2)
    b = (sigma_p**2) / (sigma_q**2)
    c = -np.log((sigma_p**2) / (sigma_q**2)) - 1
    return 0.5 * (a + b + c)


mu_p = 0.0
sigma_p = 1.0
mu_q = 1.0
sigma_q = 1.0

print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))
