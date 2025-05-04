import cv2 
import numpy as np
import os
from typing import Dict, List, Tuple, Callable, Optional, Any, Set, Union, Sequence
from cluster import clustering
from weights import update_weights, _convert_to_img_descriptor, _nearest_descriptor
from node import Node
from image import read_images, read_image, image_descriptors_map, image_descriptors, image_descriptors_from_file
from vocabulary_tree import (
    assembly_tree,
    visit_tree,
    score_calculation,
    dbimg_visit_tree,
    orb_dissimilarity,
    orb_average,
    VocabTree,
    train_voctree
)


def brute_force_image_descriptors(query_file: str,
                                 filenames: List[str],
                                 descr_extractor: cv2.Feature2D) -> Dict[float, str]:
    # Read the query image and extract its descriptors
    query_img_descriptors = image_descriptors_from_file(query_file, descr_extractor=descr_extractor)
    
    # Read the database images and extract descriptors
    images = read_images(filenames, black_white=True)
    db_descriptors_map = image_descriptors_map(images, descr_extractor=descr_extractor)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    distances: Dict[float, str] = {}
    for img_key in db_descriptors_map:
        img_descriptors = db_descriptors_map[img_key]
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




def display_comparison_table(query_files, filenames, root, db_visit_matrix, orb_extractor, orb_dissimilarity):
    
    for idx_file, query_file in enumerate(query_files): 
        print('query_file='+str(query_file))
        query_descriptors = image_descriptors_from_file(query_file, descr_extractor=orb_extractor)
        query_vector = visit_tree(root, descriptors=query_descriptors, dissimilarity=orb_dissimilarity, with_weight=True)
        
        scores = {}
        for idx, db_item_vector in enumerate(db_visit_matrix):
            scores[score_calculation(query_vector, db_item_vector)] = filenames[idx]
        
        print('voctree results:')
        idx_db = 2
        for score in sorted(scores.keys()):
            db_filename = scores[score]
            print(f"* {db_filename} {score}")
            idx_db += 1
        
        print()
        print('ground truth:')
        ground_truth = brute_force_image_descriptors(
            query_file=query_file, 
            filenames=filenames, 
            descr_extractor=orb_extractor
        )
        idx_db = 2
        for score in sorted(ground_truth.keys()):
            ground_truth_filename = ground_truth[score]
            print(f"* {ground_truth_filename} {score}")
            idx_db += 1


def main():
    filenames = list([ './imgdb/' + filename for filename in os.listdir('./imgdb/') ])
    print('database filenames')
    print('==================')
    print(str(filenames))
    print()
    
    query_files = list([ './query/' + filename for filename in os.listdir('./query/') ])
    print('query filenames')
    print('===============')
    print(str(query_files))
    
    # Set image descriptor extractor
    orb_extractor = cv2.ORB_create(750)
    
    # Number of clusters
    clusters = 3
    
    # Tree max level
    max_level = 6 
    
    # Train vocabulary tree
    voc_tree = train_voctree(
        filenames=filenames,
        image_descriptor_extractor=orb_extractor,
        clusters=clusters,
        max_level=max_level
    )
    
    # Extract root and visit_matrix from VocabTree object
    root = voc_tree.root
    db_visit_matrix = voc_tree.visit_matrix
    
    
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
        
        ground_truth = brute_force_image_descriptors(
            query_file=query_file, 
            filenames=filenames, 
            descr_extractor=orb_extractor
        )
        ground_truth_results = []
        for score in sorted(ground_truth.keys()):
            ground_truth_filename = ground_truth[score]
            ground_truth_results.append(ground_truth_filename)
            
        for idx in range(len(scores)):
            print("voctree = " + str(voctree_results[idx]))

    # Display comparison table
    display_comparison_table(query_files, filenames, root, db_visit_matrix, orb_extractor, orb_dissimilarity)

if __name__ == "__main__":
    main()