import numpy as np


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_sim = dot_product / (norm_v1 * norm_v2)
    return round(cosine_sim, 3)


v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))
