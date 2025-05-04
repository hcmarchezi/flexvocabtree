import cv2 
import numpy as np
import os
from typing import Dict, List, Tuple, Callable, Optional, Any, Set, Union, Sequence
from cluster import clustering
from weights import update_weights, _convert_to_img_descriptor, _nearest_descriptor
from node import Node
from image import read_images, read_image, image_descriptors_map, image_descriptors, image_descriptors_from_file


def tree_traversal(node: Node) -> None:
    print('node.images='+str(node.images) + ' node.children='+str(len(node.children)))
    for child_id in node.children:
        print('node.descriptor='+str(hash(child_id)))
        tree_traversal(node.children[child_id])


def assembly_tree(descriptors: np.ndarray, k: int, 
                 dissimilarity: Callable[[np.ndarray, np.ndarray], float], 
                 average: Callable[[np.ndarray], np.ndarray], 
                 level: int) -> Dict[bytes, Node]:
    level -= 1
    if level < 0:
        #print('deepest level: stop criteria')
        return {}
    elif len(descriptors) == 1:
        #print('single descriptor: stop criteria')
        key = descriptors[0].tobytes()
        return { key: Node() }

    centroids, centroid_labels = clustering(descriptors, k, dissimilarity, average)
    children: Dict[bytes, Node] = {}
    for centroid_id, centroid in enumerate(centroids):
        if len(centroid_labels[centroid_id]) > 0:
            node = Node()
            centroid_descriptors = np.array(descriptors[centroid_labels[centroid_id]])
            node.children = assembly_tree(centroid_descriptors, k, dissimilarity, average, level)
            children[centroid.tobytes()] = node
            
    return children


def visit_tree(root: Node, descriptors: np.ndarray, 
              dissimilarity: Callable[[np.ndarray, np.ndarray], float], 
              with_weight: bool = False) -> np.ndarray:
    visit_path = np.zeros(len(Node.nodes()))
    for descriptor in descriptors:
        _descriptor_visit_tree(root, descriptor, dissimilarity, visit_path)
        
    for idx, visit in enumerate(visit_path):
        visit_path[idx] /= len(descriptors) if len(descriptors) > 0 else 1
        
    if with_weight:
        for idx, visit in enumerate(visit_path):
            visit_path[idx] *= Node.nodes()[idx].weight

    return visit_path


def _descriptor_visit_tree(node: Node, descriptor: np.ndarray, 
                          dissimilarity: Callable[[np.ndarray, np.ndarray], float], 
                          visit_path: np.ndarray) -> np.ndarray:
    if node.children == {}:
        return visit_path
    
    arr_descriptors = list(map(_convert_to_img_descriptor, node.children.keys()))
    nearest_descriptor = _nearest_descriptor(descriptor, arr_descriptors, dissimilarity)
    child_node = node.children[nearest_descriptor.tobytes()]
    visit_path[child_node.index] += 1
    
    return _descriptor_visit_tree(child_node, descriptor, dissimilarity, visit_path)


# query_vector = vector of visits multiplied by weights from query image descriptors
# dbimg_vector = vector of visits multiplied by weights from database image descriptors
def score_calculation(query_vector: np.ndarray, dbimg_vector: np.ndarray) -> float:
    norm_query_vector = query_vector / (np.sqrt(np.sum(np.power(query_vector, 2))))
    norm_dbimg_vector = dbimg_vector / (np.sqrt(np.sum(np.power(query_vector, 2))))
    diff = np.abs(norm_query_vector - norm_dbimg_vector)
    return float(np.sqrt(np.sum(np.power(diff, 2))))


# root:            tree root node
#
# cvmat_images:    list of B&W cvMat images read from cv2.imread
#
# dissimilarity:   dissimilarity function to compare 2 image descriptors
#
# with_weight:     (T/F) should multiply with node weight or just number of visits per node
#
# descr_extractor: image descriptor extractor (as default ORB)
#
def dbimg_visit_tree(root: Node, cvmat_images: List[np.ndarray], 
                     dissimilarity: Callable[[np.ndarray, np.ndarray], float], 
                     with_weight: bool, descr_extractor: Any) -> np.ndarray:
    # list of visit per database image
    dbimg_vectors: List[np.ndarray] = []
    for img in cvmat_images:
        keypoints, descriptors = descr_extractor.detectAndCompute(img, None)
        if descriptors is None:
            descriptors = np.array([])
        
        # create visit vector from image and add to list
        img_vector = visit_tree(root, descriptors, dissimilarity, with_weight)
        dbimg_vectors.append(img_vector)
        
    return np.array(dbimg_vectors)


# pre-condition: uint8 arrays - hamming distance
def orb_dissimilarity(elem1: np.ndarray, elem2: np.ndarray) -> float:
    return float(cv2.norm(elem1, elem2, cv2.NORM_HAMMING))


# pre-condition: uint8 arrays - bit-wise average
def orb_average(data: np.ndarray) -> np.ndarray:
    bit_array = np.unpackbits(data, axis=1)
    avg_array = np.mean(bit_array, axis=0)
    avg_array = np.round(avg_array).astype(np.uint8)
    avg_data = np.packbits(avg_array)
    return avg_data

