import numpy as np
from collections import deque

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implement DBSCAN clustering algorithm.
    
    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - eps: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood for a core point
    
    Returns:
    - labels: 1D numpy array of cluster labels (-1 for noise points)
    """
    n_samples = X.shape[0]
    labels = -np.ones(n_samples)
    cluster_id = 0
    visited = set()

    for point_idx in range(n_samples):
        if labels[point_idx] != -1:
            continue

        neighbors = []
        for i in range(n_samples):
            if np.linalg.norm(X[point_idx] - X[i]) <= eps:
                neighbors.append(i)

        if len(neighbors) < min_samples:
            labels[point_idx] = -1
        else:
            labels[point_idx] = cluster_id
            queue = deque(neighbors)
            
            while queue:
                current_point = queue.popleft()
                
                if labels[current_point] == -1:
                    labels[current_point] = cluster_id
                elif labels[current_point] != -1:
                    continue
                
                labels[current_point] = cluster_id
                
                current_neighbors = []
                for i in range(n_samples):
                    if np.linalg.norm(X[current_point] - X[i]) <= eps:
                        current_neighbors.append(i)
                
                if len(current_neighbors) >= min_samples:
                    for neighbor_idx in current_neighbors:
                        if labels[neighbor_idx] == -1:
                            labels[neighbor_idx] = cluster_id
                            queue.append(neighbor_idx)
            
            cluster_id += 1

    return np.array(labels, dtype=int)
