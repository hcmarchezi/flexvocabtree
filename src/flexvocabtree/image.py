import cv2
import numpy as np
from typing import Dict, List, Any


def read_image(filename: str, black_white: bool = True) -> np.ndarray:
    image_mode = cv2.IMREAD_GRAYSCALE if black_white else cv2.IMREAD_COLOR
    image = cv2.imread(filename, image_mode)
    return image


def read_images(filenames: List[str], black_white: bool = True) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for filename in filenames:
        images.append(read_image(filename, black_white))
    return images


def image_descriptors_map(images: List[np.ndarray], descr_extractor: Any) -> Dict[int, np.ndarray]:
    img_descriptors: Dict[int, List[np.ndarray]] = {}

    for idx, image in enumerate(images):
        # extract image descriptors
        keypoints, descriptors = descr_extractor.detectAndCompute(image, None)
        img_descriptors[idx] = []
        if descriptors is not None:
            for descriptor in descriptors:
                img_descriptors[idx].append(descriptor)

        img_descriptors[idx] = np.array(img_descriptors[idx], dtype=np.uint8)

    return img_descriptors


def image_descriptors(image_descriptors_map: Dict[int, np.ndarray]) -> np.ndarray:
    img_descriptors: List[np.ndarray] = []
    
    for img_key in image_descriptors_map:
        img_descriptors.extend(image_descriptors_map[img_key])
        
    return np.array(img_descriptors, dtype=np.uint8)


def image_descriptors_from_file(filename: str, descr_extractor: Any) -> np.ndarray:
    img = read_image(filename)
    keypoints, descriptors = descr_extractor.detectAndCompute(img, None)
    if descriptors is None:
        descriptors = np.array([], dtype=np.uint8)
    return descriptors
