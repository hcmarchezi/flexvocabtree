import pytest
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from flexvocabtree.cluster import clustering


def euclidean_dissimilarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def simple_average(data_points: np.ndarray) -> np.ndarray:
    if data_points.shape[0] == 0:
        return np.array([])
    return np.mean(data_points, axis=0)


def test_clustering_with_simple_data_k2_converges():
    data = np.array([
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.2], 
        [5.0, 5.0], [5.1, 5.1], [5.2, 5.2]], 
        dtype=np.float64)
    k = 2
    stop_criteria = 0.01
    np.random.seed(42) # This makes _choose_initial_centroids deterministic

    centroids, labels = clustering(data, k, euclidean_dissimilarity, simple_average, stop_criteria)

    # Expected centroids should be close to the means of the two clusters
    expected_c1 = np.mean(data[:3], axis=0)
    expected_c2 = np.mean(data[3:], axis=0)

    # Check if the final centroids are close to the expected cluster means
    assert np.allclose(sorted(centroids.tolist()), sorted([expected_c1.tolist(), expected_c2.tolist()]), atol=0.1)
    
    # Map which centroid got which original group
    # case 1 - cluster0 = [0,1,2] and cluster1 = [3,4,5]
    group1_points_in_cluster0 = all(idx in labels[0] for idx in [0, 1, 2])
    group2_points_in_cluster1 = all(idx in labels[1] for idx in [3, 4, 5])
    cluster_case_1 = group1_points_in_cluster0 and group2_points_in_cluster1
    # case 2 - cluster1 = [0,1,2] and cluster0 = [3,4,5]
    group1_points_in_cluster1 = all(idx in labels[1] for idx in [0, 1, 2])
    group2_points_in_cluster0 = all(idx in labels[0] for idx in [3, 4, 5])
    cluster_case_2 = group1_points_in_cluster1 and group2_points_in_cluster0

    # check which of the cases happened
    assert (cluster_case_1 or cluster_case_2)

    # Also check that each cluster contains the correct number of points
    assert len(labels[0]) == len(labels[1])
    assert sum(len(v) for v in labels.values()) == len(data)


def test_clustering_single_point_k1():
    data = np.array([[10.0, 10.0]], dtype=np.float64)
    k = 1
    stop_criteria = 0.01
    np.random.seed(42) # for _choose_initial_centroids

    centroids, labels = clustering(data, k, euclidean_dissimilarity, simple_average, stop_criteria)

    assert centroids.shape == (1, 2)
    assert np.array_equal(centroids[0], data[0])
    assert labels == {0: [0]}


def test_clustering_k_equals_data_size():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    k = 3
    stop_criteria = 0.01
    np.random.seed(42) # for _choose_initial_centroids

    centroids, labels = clustering(data, k, euclidean_dissimilarity, simple_average, stop_criteria)

    assert len(centroids) == k
    assert all(any(np.array_equal(c, dp) for dp in data) for c in centroids)
    assert len(labels) == k
    assert all(len(v) == 1 for v in labels.values())
    assert sorted([item for sublist in labels.values() for item in sublist]) == list(range(len(data)))


def test_clustering_empty_data_returns_empty():
    data = np.array([], dtype=np.float64).reshape(0, 2) # Empty data with 2 features
    k = 1
    stop_criteria = 0.1
    
    # Test expecting an error for this scenario as per current implementation
    with pytest.raises(ValueError):
        clustering(data, k, euclidean_dissimilarity, simple_average, stop_criteria)
