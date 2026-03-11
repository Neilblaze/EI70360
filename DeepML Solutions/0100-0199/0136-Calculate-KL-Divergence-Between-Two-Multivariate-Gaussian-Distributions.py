import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    n = mu_p.shape[0]
    inverted_cov_q = np.linalg.inv(Cov_q)
    diff = mu_q - mu_p

    tr_term = np.trace(inverted_cov_q @ Cov_p)
    det_term = np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))
    quad_term = diff.T @ np.linalg.inv(Cov_q) @ diff
    return 0.5 * (tr_term + det_term + quad_term - n)
