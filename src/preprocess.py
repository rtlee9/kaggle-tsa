"""Preprocess raw a3d scan images for use in model training."""
import numpy as np
from skimage.transform import resize
from os import path
import json
import tsahelper.tsahelper as tsa
from tqdm import tqdm

from .config import path_a3d, path_cache, verbose, path_plots, path_logs
from .constants import IMAGE_DIM
from .utils import save_image, get_labels


def preprocess_image(img):
    """Scale pixel intensity, and remove mean and variance from image."""
    rescaled_img = tsa.convert_to_grayscale(img)
    normalized_img = tsa.normalize(rescaled_img)
    zero_centered_img = tsa.zero_center(normalized_img)
    return zero_centered_img


def get_upper_bound(s, sensitivity):
    """Get the upper bound given pixel sums along a given dimension."""
    d0 = s - np.roll(s, 10)  # calculate the first derivative
    d1 = d0 - np.roll(d0, 1)  # calculate the second derivative
    d1_bin = d1 < sensitivity  # binarize the second derivative
    return np.argmax(np.arange(s.shape[0]) * (d1_bin ^ np.roll(d1_bin, 1)))  # calculate the cutoff


def get_lower_bound(s, sensitivity):
    """Get the lower bound given pixel sums along a given dimension."""
    d0 = s - np.roll(s, -10)  # calculate the first derivative
    d1 = d0 - np.roll(d0, -1)  # calculate the second derivative
    d1_bin = d1 < sensitivity  # binarize the second derivative
    return np.argmax(np.arange(s.shape[0], 0, -1) * (d1_bin ^ np.roll(d1_bin, -1)))  # calculate the cutoff


def get_bounds(s, sensitivity=.4e-4, n=10):
    """Get lower and upper bounds using moving average derivative."""
    f = s * (s > sensitivity)  # remove layer-level noise
    ma = np.convolve(f, np.ones((n,)) / n, mode='valid')  # moving average
    d = ma - np.roll(ma, -n)
    _buffer = np.floor(n * .8).astype(int)
    return np.argmin(d) - _buffer, np.argmax(d) + n + _buffer


def crop_image(image):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    s0 = image.mean(axis=1).mean(axis=1)
    s1 = image.mean(axis=0).mean(axis=1)
    s2 = image.mean(axis=0).mean(axis=0)

    top_border = get_upper_bound(s0, 2.5e-7)
    right_border = get_upper_bound(s1, .9e-7)
    left_border = get_lower_bound(s1, .9e-7)
    avg_lr_border = np.floor(left_border + (s1.shape[0] - right_border) / 2).astype(int)
    front_border, back_border = get_bounds(s2)

    resized_image = image[:top_border, avg_lr_border:s1.shape[0] - avg_lr_border, front_border:back_border]
    if verbose > 1:
        print('Image resized from {} to {}'.format(image.shape, resized_image.shape))
    return resized_image


def preprocess_tsa_data(type='labels'):
    """Preprocess all a3d files for training and persist to disk."""
    scans = get_labels(type)
    crop_log = {}
    for subject_id in tqdm(scans.subject_id.unique()):
        image = tsa.read_data(path.join(path_a3d, subject_id + '.a3d'))
        image = image.transpose(2, 0, 1)  # axis are now height (top) x width (side) x  depth (front)
        cropped_image = crop_image(image)
        preprocessed_image = preprocess_image(cropped_image)
        resized_image = resize(preprocessed_image, (IMAGE_DIM, IMAGE_DIM, IMAGE_DIM), mode='constant')
        np.save(path.join(path_cache, subject_id + '.npy'), resized_image)
        crop_log[subject_id] = cropped_image.shape

        # save a cross section for validating cropping
        save_image(
            path.join(path_plots, subject_id + '.png'),
            tsa.convert_to_grayscale(resized_image)[:, :, np.floor(resized_image.shape[2] / 2).astype(int) - 5],
        )

    with open(path.join(path_logs, 'crop_log_{}.json'.format(type)), 'w') as f:
        json.dump(crop_log, f, indent=4)


if __name__ == '__main__':
    preprocess_tsa_data()
    preprocess_tsa_data('submissions')
