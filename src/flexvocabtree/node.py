import numpy as np
from typing import Dict, Set, List


class Node:
    _nodes: List['Node'] = []       # list of all created nodes
    _total_images: int = 0         # total number of images
    
    def __init__(self, root: bool = False) -> None:
        self._children: Dict[bytes, 'Node'] = {}  # dictionary key(img descriptor) -> value
        self._images: Set[int] = set()            # set of image ids
        if not root:
            self._index: int = len(Node._nodes)
            Node._nodes.append(self) # append node to global list of nodes

    @property
    def children(self) -> Dict[bytes, 'Node']:
        return self._children

    @children.setter
    def children(self, children: Dict[bytes, 'Node']) -> None:
        self._children = children

    @property
    def images(self) -> Set[int]:
        return self._images

    @images.setter
    def images(self, images: Set[int]) -> None:
        self._images = images
        
    @property
    def weight(self) -> float:
        images_in_node = len(self._images)
        if images_in_node > 0:
            return np.log(Node._total_images/len(self._images))
        else:
            return 0
        
    @property
    def index(self) -> int:
        return self._index
    
    @staticmethod
    def set_total_images(total_images: int) -> None:
        Node._total_images = total_images
        
    @staticmethod
    def nodes() -> List['Node']:
        return Node._nodes
    
    @staticmethod
    def init() -> None:
        Node._nodes = []
