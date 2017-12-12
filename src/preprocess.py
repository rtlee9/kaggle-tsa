"""Preprocess raw a3d scan images for use in model training."""
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
from os import path
import json
import tsahelper.tsahelper as tsa
from tqdm import tqdm

from .config import path_a3d, path_cache, verbose, path_plots, path_logs, stage
from .constants import IMAGE_DIM
from .utils import save_image, get_labels, moving_average, plot_line


def find_edges(a, buffer=0, plot_distr=False):
    """Find the edges of a series."""
    ma = moving_average(a, 10)
    f = ma > 10000

    if plot_distr:
        plot_line(ma)

    lower, upper = np.argmax(f), f.shape[0] - np.argmax(f[::-1])
    if f[0]:
        lower = 0
    if f[-1]:
        upper = f.shape[0]
    return max(lower - buffer, 0), min(upper + buffer, f.shape[0])


def crop_image(image, buffer=0):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    convolved = convolve(image, np.ones((2, 2, 2)))
    filtered = (convolved * (convolved > 250))

    s0 = filtered.sum(axis=1).sum(axis=1)
    s1 = filtered.sum(axis=0).sum(axis=1)
    s2 = filtered.sum(axis=0).sum(axis=0)

    # borders for each dimension
    bottom, top = find_edges(s0, buffer)
    left, right = find_edges(s1, buffer)
    front, back = find_edges(s2, buffer)

    resized_image = image[:top, left:right, front:back]
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
        image = tsa.convert_to_grayscale(image)
        cropped_image = crop_image(image)
        resized_image = resize(cropped_image, (IMAGE_DIM, IMAGE_DIM, IMAGE_DIM), mode='constant')
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
    if stage == 1:
        preprocess_tsa_data()
        preprocess_tsa_data('submissions')
    if stage == 2:
        preprocess_tsa_data('submissions2')
