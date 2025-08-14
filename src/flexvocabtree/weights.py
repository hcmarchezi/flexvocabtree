import numpy as np
from typing import List, Optional, Callable
from flexvocabtree.node import Node


def update_weights(
        node: Node, img_idx: int, arr_descriptors: np.ndarray,
        dissimilarity: Callable[[np.ndarray, np.ndarray], float]) -> None:
    for descriptor in arr_descriptors:
        _update_weights_with_descriptor(node, img_idx, descriptor, dissimilarity)


# descriptor - query descriptor
# arr_descriptors - array of descriptors
# return: descritor in arr_descriptors close to descriptor
def _nearest_descriptor(descriptor: np.ndarray, arr_descriptors: List[np.ndarray],
                        dissimilarity: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    min_dist: Optional[float] = None
    min_descriptor: Optional[np.ndarray] = None
    for descriptor_item in arr_descriptors:
        distance = dissimilarity(descriptor, descriptor_item)
        if min_dist is None or distance < min_dist:
            min_dist = distance
            min_descriptor = descriptor_item
    # Add fallback to first descriptor if none matched (should not happen in practice)
    if min_descriptor is None and len(arr_descriptors) > 0:
        min_descriptor = arr_descriptors[0]
    assert min_descriptor is not None, "No descriptors provided"
    return min_descriptor


def _convert_to_img_descriptor(descriptor_bytes: bytes) -> np.ndarray:
    return np.frombuffer(descriptor_bytes, dtype=np.uint8)


def _update_weights_with_descriptor(
        node: Node, img_idx: int, descriptor: np.ndarray,
        dissimilarity: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    if len(node.children) > 0:
        arr_descriptors = list(map(_convert_to_img_descriptor, node.children.keys()))
        nearest_descriptor = _nearest_descriptor(descriptor, arr_descriptors, dissimilarity)
        child_node = node.children[nearest_descriptor.tobytes()]
        child_node.images.add(img_idx)
        _update_weights_with_descriptor(child_node, img_idx, descriptor, dissimilarity)
