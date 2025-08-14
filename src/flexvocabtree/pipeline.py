import cv2
import numpy as np
from typing import Dict, Callable
from flexvocabtree.vocabulary_tree import VocabTree, visit_tree, score_calculation
from flexvocabtree.image import image_descriptors_from_file


class Pipeline:
    def __init__(
            self,
            vocab_tree: VocabTree,
            descr_extractor: cv2.Feature2D,
            dissimilarity: Callable[[np.ndarray, np.ndarray], float]
    ):
        self._vocab_tree = vocab_tree
        self._descr_extractor = descr_extractor
        self._dissimilarity = dissimilarity

    def execute(self, query_file: str) -> Dict[int, str]:
        query_descriptors = image_descriptors_from_file(query_file, descr_extractor=self._descr_extractor)
        query_vector = visit_tree(
            self._vocab_tree.root,
            descriptors=query_descriptors,
            dissimilarity=self._dissimilarity,
            with_weight=True)

        scores = {}
        for idx, db_item_vector in enumerate(self._vocab_tree.visit_matrix):
            scores[score_calculation(query_vector, db_item_vector)] = self._vocab_tree.database_filenames[idx]

        return scores
