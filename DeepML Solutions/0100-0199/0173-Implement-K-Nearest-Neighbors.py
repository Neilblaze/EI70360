import math

def k_nearest_neighbors(points, query_point, k):
    euclidean_distance = lambda p1, p2: math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    sorted_points = sorted(points, key=lambda p: euclidean_distance(p, query_point))
    return sorted_points[:k]
