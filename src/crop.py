"""Crop utility functions."""
import numpy as np


def hard_crop(image, crop_boundaries):
    """Crop image based on pixel boundaries."""
    # input sanity checks
    expected_dims = np.array([(boundary[1] - boundary[0]) for boundary in crop_boundaries])
    assert len(crop_boundaries) == 3
    assert np.all(expected_dims > 0)

    # check input has has three sets of two boundaries
    for boundary_set in crop_boundaries:
        for boundary in boundary_set:
            assert boundary % 1 == 0, boundary

    for boundary_set in crop_boundaries:
        assert(len(boundary_set)) == 2

    scale = image.shape[0] / 128
    crop_boundaries = [
        [int(crop_edge * scale) for crop_edge in crop_dim]
        for crop_dim in crop_boundaries]
    cropped_image = image[
        crop_boundaries[0][0]:crop_boundaries[0][1],
        crop_boundaries[1][0]:crop_boundaries[1][1],
        crop_boundaries[2][0]:crop_boundaries[2][1],
    ]
    assert sum(cropped_image.shape) == expected_dims.sum()
    return cropped_image


def test_hard_crop():
    dummy_image = np.zeros((100, 100, 100))
    crop_boundaries = ((45, 80), (15, 20), (29, 41))
    assert hard_crop(dummy_image, crop_boundaries).shape == (35, 5, 12)


def crop_dims(image_dims, top=None, bottom=None, left=None, right=None, front=None, back=None):
    """Get crop dimensions from base image dimensions and additional constraints."""
    cropped_dims = [[0, image_dims[0]], [0, image_dims[1]], [0, image_dims[2]]]
    if bottom:
        cropped_dims[0][0] = bottom
    if top:
        cropped_dims[0][1] = top
    if left:
        cropped_dims[1][0] = left
    if right:
        cropped_dims[1][1] = right
    if front:
        cropped_dims[2][0] = front
    if back:
        cropped_dims[2][1] = back
    return cropped_dims


class TestCropDims(object):
    def test_crop_dims_top(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, top=5) == [[5, 128], [0, 128], [0, 128]]

    def test_crop_dims_bottom(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, bottom=5) == [[0, 5], [0, 128], [0, 128]]

    def test_crop_dims_topandbottom(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, top=10, bottom=100) == [[10, 100], [0, 128], [0, 128]]

    def test_crop_dims_left(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, left=5) == [[0, 128], [5, 128], [0, 128]]

    def test_crop_dims_right(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, right=5) == [[0, 128], [0, 5], [0, 128]]

    def test_crop_dims_leftandright(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, left=10, right=100) == [[0, 128], [10, 100], [0, 128]]

    def test_crop_dims_front(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, front=5) == [[0, 128], [0, 128], [5, 128]]

    def test_crop_dims_back(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, back=5) == [[0, 128], [0, 128], [0, 5]]

    def test_crop_dims_frontandback(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, front=10, back=100) == [[0, 128], [0, 128], [10, 100]]

    def test_crop_dims_leftandback(self):
        image_dims = (128, 128, 128)
        assert crop_dims(image_dims, left=10, back=100) == [[0, 128], [10, 128], [0, 100]]
