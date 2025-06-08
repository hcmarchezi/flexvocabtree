import cv2 
import numpy as np
import os
from typing import Dict, List, Tuple, Callable, Optional, Any, Set, Union, Sequence
from cluster import clustering
from weights import update_weights, _convert_to_img_descriptor, _nearest_descriptor
from node import Node
from image import read_images, image_descriptors_map, image_descriptors, image_descriptors_from_file
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
from pipeline import Pipeline
from columnar import columnar



def brute_force_image_descriptors(query_file: str, filenames: List[str], descr_extractor: cv2.Feature2D) -> Dict[float, str]:
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
    
    # Train vocabulary tree
    voc_tree = train_voctree(
        filenames=filenames,
        image_descriptor_extractor=orb_extractor,
        clusters=3,
        max_level=6
    )
    
    # Create pipeline to process queries
    pipeline = Pipeline(
        vocab_tree = voc_tree,
        descr_extractor = orb_extractor,
        dissimilarity = orb_dissimilarity
    )
    
    
    for idx_file, query_file in enumerate(query_files): 
        scores = pipeline.execute(query_file)    
        voctree_results = [ scores[score] for score in sorted(scores.keys()) ]
        
        ground_truth = brute_force_image_descriptors(query_file=query_file, filenames=filenames, descr_extractor=orb_extractor)
        ground_truth_results = [ ground_truth[score] for score in sorted(ground_truth.keys()) ]

        data = []
        for idx in range(len(voctree_results)):
            data.append([ query_file, voctree_results[idx], ground_truth_results[idx] ])
        table = columnar(data, headers=['Query File', 'VocabTree', 'GroundTruth'])
        print(table)



if __name__ == "__main__":
    main()
