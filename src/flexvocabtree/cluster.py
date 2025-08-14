import numpy as np
from typing import Dict, List, Tuple, Callable, Optional


# clustering image descriptors
def clustering(
        data: np.ndarray, k: int,
        dissimilarity: Callable[[np.ndarray, np.ndarray], float],
        average: Callable[[np.ndarray], np.ndarray],
        stop_criteria: float = 0.1, attempts: int = 3
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    centroids = _choose_initial_centroids(data, k)
    centroid_labels: Dict[int, List[int]] = {}
    diff = 100000.0

    while diff > stop_criteria:
        centroid_labels = _find_nearest_centroid(data, centroids, dissimilarity)
        new_centroids = _calculate_new_centroids(data, centroid_labels, k, average, centroids)
        diff = _average_centroids_move(centroids, new_centroids)
        centroids = new_centroids

    return centroids, centroid_labels


def _choose_initial_centroids(data: np.ndarray, k: int) -> np.ndarray:
    centroid_idxs = np.random.randint(data.shape[0], size=k)
    return data[centroid_idxs]


def _find_nearest_centroid(
        data: np.ndarray,
        centroids: np.ndarray,
        dissimilarity: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[int, List[int]]:
    centroid_labels: Dict[int, List[int]] = {}
    for idx in range(len(centroids)):
        centroid_labels[idx] = []

    for item_idx, item in enumerate(data):
        min_dist: Optional[float] = None
        centroid_id = 0  # Default to first centroid
        for centroid_idx, c in enumerate(centroids):
            distance = dissimilarity(item, c)
            if min_dist is None or distance < min_dist:
                min_dist = distance
                centroid_id = centroid_idx
        centroid_labels[centroid_id].append(item_idx)

    return centroid_labels


def _calculate_new_centroids(
        data: np.ndarray,
        centroid_labels: Dict[int, List[int]],
        k: int,
        average: Callable[[np.ndarray], np.ndarray],
        original_centroids: np.ndarray
) -> np.ndarray:
    centroids: List[np.ndarray] = []

    for centroid_idx in centroid_labels:
        if len(centroid_labels[centroid_idx]) == 0:
            new_centroid = original_centroids[centroid_idx]
        else:
            new_centroid = average(data[centroid_labels[centroid_idx]])
        centroids.append(new_centroid)

    return np.array(centroids)


def _average_centroids_move(centroids: np.ndarray, new_centroids: np.ndarray) -> float:
    return float(np.sum(np.abs(centroids - new_centroids)))
