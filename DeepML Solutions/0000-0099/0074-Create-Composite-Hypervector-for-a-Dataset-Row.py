import numpy as np


def create_row_hv(row, dim, random_seeds):
    hvs = []
    for col, seed in random_seeds.items():
        np.random.seed(seed)
        hv1 = np.random.choice([-1, 1], dim)
        hv2 = np.random.choice([-1, 1], dim)
        bind = hv1 * hv2
        hvs.append(bind)

    bundled = np.sum(hvs, axis=0)
    return np.where(bundled >= 0, 1, -1)


row = {"FeatureA": "value1", "FeatureB": "value2"}
dim = 5
random_seeds = {"FeatureA": 42, "FeatureB": 7}
hvc = create_row_hv(row, dim, random_seeds)
print(hvc)
