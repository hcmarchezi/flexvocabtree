import cv2
import numpy as np
from typing import Dict, List, Callable, Any
from flexvocabtree.cluster import clustering
from flexvocabtree.weights import update_weights, _convert_to_img_descriptor, _nearest_descriptor
from flexvocabtree.node import Node
from flexvocabtree.image import read_images, image_descriptors_map, image_descriptors


class VocabTree:
    """
    Vocabulary Tree for image retrieval
    Attributes:
        root: The root node of the vocabulary tree
        visit_matrix: The database image visit matrix
    """

    def __init__(self, root: Node, visit_matrix: np.ndarray, database_filenames: List[str]):
        """
        Initialize a vocabulary tree
        Args:
            root: The root node of the vocabulary tree
            visit_matrix: The database image visit matrix
            database_filenames: Database filenames
        """
        self._root = root
        self._visit_matrix = visit_matrix
        self._database_filenames = database_filenames

    @property
    def root(self) -> Node:
        """Get the root node"""
        return self._root

    @property
    def visit_matrix(self) -> np.ndarray:
        """Get the database image visit matrix"""
        return self._visit_matrix

    @property
    def database_filenames(self) -> List[str]:
        return self._database_filenames


def train_voctree(
        filenames: List[str],
        image_descriptor_extractor: cv2.Feature2D,
        clusters: int,
        max_level: int
) -> VocabTree:
    """
    Trains vocabulary tree based on input images
    Args:
        filenames: List of image file paths
        image_descriptor_extractor: OpenCV feature extractor (e.g., ORB, SIFT)
        clusters: Number of clusters for tree branching
        max_level: Maximum depth of the vocabulary tree
    Returns:
        VocabTree object containing the root node and database visit matrix
    """
    # Initialize node system
    Node.init()

    # Set number of images
    Node.set_total_images(len(filenames))

    # Read all images and extract features
    images = read_images(filenames, black_white=True)
    print('number of images=' + str(len(images)))

    # Extract descriptors from database images
    db_descriptors_map = image_descriptors_map(images, descr_extractor=image_descriptor_extractor)
    db_descriptors = image_descriptors(db_descriptors_map)
    print('number of descriptors=' + str(db_descriptors.shape[0]))

    # Create root node
    root = Node(root=True)

    # Build tree structure
    root.children = assembly_tree(
        descriptors=db_descriptors,
        k=clusters,
        dissimilarity=orb_dissimilarity,
        average=orb_average,
        level=max_level
    )
    print('number of nodes:' + str(len(Node.nodes())))

    # Update weights in the tree
    for imgkey in db_descriptors_map:
        print("update weights for image = " + str(imgkey))
        update_weights(root, imgkey, db_descriptors_map[imgkey], dissimilarity=orb_dissimilarity)

    # Create database visit matrix
    db_visit_matrix = dbimg_visit_tree(
        root,
        images,
        dissimilarity=orb_dissimilarity,
        with_weight=True,
        descr_extractor=image_descriptor_extractor
    )
    print('database image vector size = ' + str(db_visit_matrix.shape))

    return VocabTree(root=root, visit_matrix=db_visit_matrix, database_filenames=filenames)


def tree_traversal(node: Node) -> None:
    print('node.images=' + str(node.images) + ' node.children=' + str(len(node.children)))
    for child_id in node.children:
        print('node.descriptor=' + str(hash(child_id)))
        tree_traversal(node.children[child_id])


def assembly_tree(descriptors: np.ndarray, k: int,
                  dissimilarity: Callable[[np.ndarray, np.ndarray], float],
                  average: Callable[[np.ndarray], np.ndarray],
                  level: int) -> Dict[bytes, Node]:
    level -= 1
    if level < 0:
        return {}
    elif len(descriptors) == 1:
        key = descriptors[0].tobytes()
        return {key: Node()}

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

    tf_normalized_visits = _apply_tf_normalization(visit_path)

    if with_weight:
        # Apply IDF weighting to TF-normalized visits (TF-IDF)
        for idx, tf_visit in enumerate(tf_normalized_visits):
            tf_normalized_visits[idx] *= Node.nodes()[idx].weight

    return tf_normalized_visits


def _apply_tf_normalization(visit_path: np.ndarray) -> np.ndarray:
    tf_normalized = np.zeros_like(visit_path, dtype=np.float64)
    
    for idx, count in enumerate(visit_path):
        if count > 0:
            # Log normalization: 1 + log(count)
            tf_normalized[idx] = 1.0 + np.log(count)
        else:
            tf_normalized[idx] = 0.0
    
    return tf_normalized


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
    norm_query_vector = query_vector / np.linalg.norm(query_vector, ord=2)
    norm_dbimg_vector = dbimg_vector / np.linalg.norm(dbimg_vector, ord=2)
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
