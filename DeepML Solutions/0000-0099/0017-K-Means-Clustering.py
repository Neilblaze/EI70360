import numpy as np


def k_means_clustering(
    points: list[tuple[float, float]],
    k: int,
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:  # return final_centroids
    points = np.array(points)
    centroids = np.array(initial_centroids)
    iteration = 0

    while iteration < max_iterations:
        distances = np.zeros((points.shape[0], k))

        # Mesafeleri hesapla
        for i in range(k):
            for j in range(points.shape[0]):
                distances[j, i] = np.linalg.norm(points[j] - centroids[i])

        labels = np.argmin(distances, axis=1)
        new_centroids = []

        for i in range(k):
            if np.any(labels == i):
                new_centroids.append(points[labels == i].mean(axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
        iteration += 1

    return [tuple(np.round(centroid, 4)) for centroid in centroids]


points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
k = 2
initial_centroids = [(1, 1), (10, 1)]
max_iterations = 10
final_centroids = k_means_clustering(points, k, initial_centroids, max_iterations)
print(final_centroids)
