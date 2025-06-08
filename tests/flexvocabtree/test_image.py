import pytest
import cv2
import numpy as np
from flexvocabtree.image import read_image, read_images, image_descriptors_map, image_descriptors, image_descriptors_from_file


@pytest.fixture
def mock_cv2_imread(mocker):
    """Fixture to mock cv2.imread."""
    mock_imread = mocker.patch('cv2.imread')
    mock_imread.return_value = np.zeros((10, 10), dtype=np.uint8)  # Default return
    return mock_imread

@pytest.fixture
def mock_descr_extractor(mocker):
    """Fixture to mock a descriptor extractor."""
    mock_extractor = mocker.Mock()
    mock_extractor.detectAndCompute.return_value = ([], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
    return mock_extractor


def test_read_image_black_white(mock_cv2_imread):
    filename = "test_bw.png"
    image = read_image(filename, black_white=True)
    mock_cv2_imread.assert_called_once_with(filename, cv2.IMREAD_GRAYSCALE)
    assert isinstance(image, np.ndarray)


def test_read_image_color(mock_cv2_imread):
    filename = "test_color.jpg"
    image = read_image(filename, black_white=False)
    mock_cv2_imread.assert_called_once_with(filename, cv2.IMREAD_COLOR)
    assert isinstance(image, np.ndarray)


def test_read_images_black_white(mocker):    
    mocker.patch('flexvocabtree.image.read_image', return_value=np.zeros((20, 20), dtype=np.uint8))
    
    filenames = ["img1.png", "img2.png"]
    images = read_images(filenames, black_white=True)
    
    assert len(images) == len(filenames)
    assert all(isinstance(img, np.ndarray) for img in images)
    for filename in filenames:
        mocker.call.read_image(filename, True).assert_called_once() 


def test_read_images_color(mocker):
    mocker.patch('flexvocabtree.image.read_image', return_value=np.zeros((30, 30, 3), dtype=np.uint8))

    filenames = ["imgA.jpg", "imgB.jpg"]    
    images = read_images(filenames, black_white=False)

    assert len(images) == len(filenames)
    assert all(isinstance(img, np.ndarray) for img in images)
    for filename in filenames:
        mocker.call.read_image(filename, False).assert_called_once() 


def test_image_descriptors_map_with_descriptors(mock_descr_extractor):
    mock_descr_extractor.detectAndCompute.side_effect = [
        ([], np.array([[10, 11], [12, 13]], dtype=np.uint8)),
        ([], np.array([[20, 21, 22]], dtype=np.uint8))
    ]

    images = [np.zeros((100, 100), dtype=np.uint8), np.zeros((50, 50), dtype=np.uint8)]
    descriptors_map = image_descriptors_map(images, mock_descr_extractor)
    
    assert tuple(descriptors_map.keys()) == (0, 1)
    assert np.array_equal(descriptors_map[0], np.array([[10, 11], [12, 13]], dtype=np.uint8))
    assert np.array_equal(descriptors_map[1], np.array([[20, 21, 22]], dtype=np.uint8))
    mock_descr_extractor.detectAndCompute.call_count == len(images)


def test_image_descriptors_map_no_descriptors(mock_descr_extractor):
    mock_descr_extractor.detectAndCompute.return_value = ([], None)

    images = [np.zeros((100, 100), dtype=np.uint8)]
    descriptors_map = image_descriptors_map(images, mock_descr_extractor)
    
    assert tuple(descriptors_map.keys()) == (0,)
    assert descriptors_map[0].size == 0 
    assert descriptors_map[0].dtype == np.uint8


def test_image_descriptors():    
    map_input_consistent_dim = {
        0: np.array([[1, 2], [3, 4]], dtype=np.uint8),
        1: np.array([[5, 6], [7, 8]], dtype=np.uint8) # Changed to 2 columns
    }

    all_descriptors_consistent_dim = image_descriptors(map_input_consistent_dim)

    expected_output_consistent_dim = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)
    assert np.array_equal(all_descriptors_consistent_dim, expected_output_consistent_dim)
    assert all_descriptors_consistent_dim.dtype == np.uint8


def test_image_descriptors_empty_map():
    map_input = {}
    all_descriptors = image_descriptors(map_input)
    assert all_descriptors.size == 0
    assert all_descriptors.dtype == np.uint8


def test_image_descriptors_from_file_with_descriptors(mocker, mock_descr_extractor):
    filename = "test_file.png"
    mock_read_image = mocker.patch('flexvocabtree.image.read_image', return_value=np.zeros((100, 100), dtype=np.uint8))
    mock_descr_extractor.detectAndCompute.return_value = ([], np.array([[10, 20]], dtype=np.uint8))
    
    descriptors = image_descriptors_from_file(filename, mock_descr_extractor)
    
    mock_read_image.assert_called_once_with(filename)
    mock_descr_extractor.detectAndCompute.assert_called_once()
    assert np.array_equal(descriptors, np.array([[10, 20]], dtype=np.uint8))


def test_image_descriptors_from_file_no_descriptors(mocker, mock_descr_extractor):
    filename = "no_descr.png"
    mock_read_image = mocker.patch('flexvocabtree.image.read_image', return_value=np.zeros((50, 50), dtype=np.uint8))
    mock_descr_extractor.detectAndCompute.return_value = ([], None)
    
    descriptors = image_descriptors_from_file(filename, mock_descr_extractor)
    
    mock_read_image.assert_called_once_with(filename)
    mock_descr_extractor.detectAndCompute.assert_called_once()
    assert np.array_equal(descriptors, np.array([], dtype=np.uint8))

