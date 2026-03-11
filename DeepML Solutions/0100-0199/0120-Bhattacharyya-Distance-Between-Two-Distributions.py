import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) == len(q):
        return round(-np.log(sum([np.sqrt(a * b) for a, b in zip(p, q)])), 4)
    else:
        return 0.0

p = [0.1, 0.2, 0.3, 0.4]
q = [0.4, 0.3, 0.2, 0.1]
result = bhattacharyya_distance(p, q)
print(result)