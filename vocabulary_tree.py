import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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




def brute_force_image_descriptors(query_img_descriptors: np.ndarray, 
                                 image_descriptors_map: Dict[int, np.ndarray], 
                                 filenames: List[str]) -> Dict[float, str]:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    distances: Dict[float, str] = {}
    for img_key in image_descriptors_map:
        img_descriptors = image_descriptors_map[img_key]
        if img_descriptors.shape[0] > 0 and query_img_descriptors.shape[0] > 0:
            matches = bf.match(query_img_descriptors, img_descriptors) 
            dist_values = [float(item.distance) for item in matches]
            dist_values = sorted(dist_values)
            mean_dist = float(np.mean(dist_values))
            distances[mean_dist] = filenames[img_key]
        else:
            print("img_descriptors empty for image = " + str(img_key) + " set a high distance!")
            distances[10000.0] = filenames[img_key]
        
    return distances


## TESTS

filenames = list([ './imgdb/' + filename for filename in os.listdir('./imgdb/') ])
print('database filenames')
print('==================')
print(str(filenames))
print()

query_files = list([ './query/' + filename for filename in os.listdir('./query/') ])
print('query filenames')
print('===============')
print(str(query_files))


## Test1: ORB / hamming distance / bit-wise average

# pre-condition: uint8 arrays - hamming distance
def orb_dissimilarity(elem1: np.ndarray, elem2: np.ndarray) -> float:
    return float(cv2.norm(elem1, elem2, cv2.NORM_HAMMING))

# # pre-condition: uint8 arrays - bit-wise average
def orb_average(data: np.ndarray) -> np.ndarray:
    bit_array = np.unpackbits(data, axis=1)
    avg_array = np.mean(bit_array, axis=0)
    avg_array = np.round(avg_array).astype(np.uint8)
    avg_data = np.packbits(avg_array)
    return avg_data

# Set number of images
Node.set_total_images(len(filenames))

# Set image descriptor extractor
orb_extractor = cv2.ORB_create(750)

# Number of clusters
clusters = 3

# Tree max level
max_level = 6 

images = read_images(filenames, black_white=True)
print('number of images='+str(len(images)))

db_descriptors_map = image_descriptors_map(images, descr_extractor=orb_extractor)
db_descriptors = image_descriptors(db_descriptors_map)
print('number of descriptors='+str(db_descriptors.shape[0]))

Node.init()
root = Node(root=True)
root.children = assembly_tree(descriptors=db_descriptors, k=clusters, dissimilarity=orb_dissimilarity, average=orb_average, level=max_level)
print('number of nodes:' + str(len(Node.nodes())))

for imgkey in db_descriptors_map:
    print("update weights for image = " + str(imgkey))
    update_weights(root, imgkey, db_descriptors_map[imgkey], dissimilarity=orb_dissimilarity)

db_visit_matrix = dbimg_visit_tree(root, images, dissimilarity=orb_dissimilarity, with_weight=True, descr_extractor=orb_extractor)
print('database image vector size = '+str(db_visit_matrix.shape))


for idx_file, query_file in enumerate(query_files): 
    print('----------------------------------------------')
    print('query_file='+str(query_file))
    print('----------------------------------------------')
    query_descriptors = image_descriptors_from_file(query_file, descr_extractor=orb_extractor)
    query_vector = visit_tree(root, descriptors=query_descriptors, dissimilarity=orb_dissimilarity, with_weight=True)
    
    scores = {}
    for idx, db_item_vector in enumerate(db_visit_matrix):
        scores[score_calculation(query_vector, db_item_vector)] = filenames[idx]
    
    voctree_results = []
    for score in sorted(scores.keys()):
        db_filename = scores[score]
        voctree_results.append(db_filename)
    
    ground_truth = brute_force_image_descriptors(query_descriptors, db_descriptors_map, filenames)    
    ground_truth_results = []
    for score in sorted(ground_truth.keys()):
        ground_truth_filename = ground_truth[score]
        ground_truth_results.append(ground_truth_filename)
        
    for idx in range(len(scores)):
        print("voctree = " + str(voctree_results[idx]))

## Display comparison table

fig, axarr = plt.subplots(nrows = 2 + len(filenames), ncols = len(query_files)*3, figsize=(20,55))

for idx_file, query_file in enumerate(query_files): 
    print('query_file='+str(query_file))
    query_descriptors = image_descriptors_from_file(query_file, descr_extractor=orb_extractor)
    query_vector = visit_tree(root, descriptors=query_descriptors, dissimilarity=orb_dissimilarity, with_weight=True)
    
    scores = {}
    for idx, db_item_vector in enumerate(db_visit_matrix):
        scores[score_calculation(query_vector, db_item_vector)] = filenames[idx]
    
    axarr[0, idx_file*3].imshow(read_image(query_file), cmap='gray')
    axarr[0, idx_file*3].set_title(query_file)
    axarr[0, idx_file*3].axis('off')
    
    axarr[0, idx_file*3 + 1].axis('off')
    axarr[0, idx_file*3 + 2].axis('off')
    
    axarr[1, idx_file*3].axis('off')
    axarr[1, idx_file*3 + 1].axis('off')
    axarr[1, idx_file*3 + 2].axis('off')
    
    axarr[1, idx_file*3].set_title('voctree')  
    axarr[1, idx_file*3 + 1].set_title('ground truth')
    
    idx_db = 2
    for score in sorted(scores.keys()):
        db_filename = scores[score]
        axarr[idx_db, idx_file*3].imshow(read_image(db_filename), cmap='gray')
        axarr[idx_db, idx_file*3].set_title(round(score, 2))
        axarr[idx_db, idx_file*3].set_xticklabels([])
        axarr[idx_db, idx_file*3].set_yticklabels([])
        idx_db += 1
    
    ground_truth = brute_force_image_descriptors(query_descriptors, db_descriptors_map, filenames)    
    idx_db = 2
    for score in sorted(ground_truth.keys()):
        ground_truth_filename = ground_truth[score]
        axarr[idx_db, idx_file*3 + 1].imshow(read_image(ground_truth_filename), cmap='gray')
        axarr[idx_db, idx_file*3 + 1].set_title(round(score,2))
        axarr[idx_db, idx_file*3 + 1].set_xticklabels([])
        axarr[idx_db, idx_file*3 + 1].set_yticklabels([])
        
        axarr[idx_db, idx_file*3 + 2].axis('off')
        
        idx_db += 1
        
plt.show()









