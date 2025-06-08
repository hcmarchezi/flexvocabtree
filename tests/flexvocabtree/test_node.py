import numpy as np
from flexvocabtree.node import Node


def test_create_root_node():
	root = Node(root = True)
	assert root.children == {}
	assert root.images == set()

def test_create_non_root_node():
	Node.init()
	node = Node()
	assert node.children == {}
	assert node.images == set()
	assert Node.nodes() == [node]

def test_images_property():
	node = Node()
	node.images = {1,10,11,20,25,30,50,100}
	assert node.images == {1,10,11,20,25,30,50,100}

def test_children_properties():
	child_1 = Node()
	child_2 = Node()
	child_3 = Node()

	node = Node()
	node.children = {b'100': child_1, b'200': child_2, b'300': child_3}

	assert node.children == {b'100': child_1, b'200': child_2, b'300': child_3}

def test_weight():
	Node.set_total_images(10)
	node = Node()
	node.images = {1,2}
	assert node.weight == np.log(10/2)
