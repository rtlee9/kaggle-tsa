"""Preprocess raw a3d scan images for use in model training."""
import numpy as np
import pandas as pd
from skimage.transform import resize
from os import path
import json
import tsahelper.tsahelper as tsa
from tqdm import tqdm

from .config import path_labels, path_a3d, path_cache, verbose, path_plots
from .constants import IMAGE_DIM
from .utils import save_image


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


def crop_image(image):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    s0 = image.mean(axis=1).mean(axis=1)
    s1 = image.mean(axis=0).mean(axis=1)
    s2 = image.mean(axis=0).mean(axis=0)

    top_border = get_upper_bound(s0, 2.5e-7)
    right_border = get_upper_bound(s1, .9e-7)
    left_border = get_lower_bound(s1, .9e-7)
    avg_lr_border = np.floor(left_border + (s1.shape[0] - right_border) / 2).astype(int)
    back_border = get_upper_bound(s2, .5e-6)
    front_border = get_lower_bound(s2, .5e-6)

    resized_image = image[:top_border, avg_lr_border:s1.shape[0] - avg_lr_border, front_border:back_border]
    if verbose > 1:
        print('Image resized from {} to {}'.format(image.shape, resized_image.shape))
    return resized_image


def preprocess_tsa_data():
    """Preprocess all a3d files for training and persist to disk."""
    # get list of scans
    scans = pd.read_csv(path_labels)
    scans['subject_id'] = scans.Id.str.split('_').str[0]
    scans['zone_num'] = scans.Id.str.split('Zone').str[1].astype(int)

    # preprocess each scan
    blacklist = {}
    for subject_id in tqdm(scans.subject_id.unique()):
        image = tsa.read_data(path.join(path_a3d, subject_id + '.a3d'))
        image = image.transpose(2, 0, 1)  # axis are now height (top) x width (side) x  depth (front)
        cropped_image = crop_image(image)
        if cropped_image.shape[0] < 550 or cropped_image.shape[1] < 300 or cropped_image.shape[1] > 520 or cropped_image.shape[2] < 100 or cropped_image.shape[2] > 300:
            blacklist[subject_id] = cropped_image.shape
            with open('blacklist.json', 'w') as f:
                json.dump(blacklist, f, indent=4)
        preprocessed_image = preprocess_image(cropped_image)
        resized_image = resize(preprocessed_image, (IMAGE_DIM, IMAGE_DIM, IMAGE_DIM), mode='constant')
        np.save(path.join(path_cache, subject_id + '.npy'), resized_image)

        # save a cross section for validating cropping
        save_image(
            path.join(path_plots, subject_id + '.png'),
            tsa.convert_to_grayscale(resized_image)[:, :, np.floor(resized_image.shape[2] / 2).astype(int) - 5],
        )


if __name__ == '__main__':
    preprocess_tsa_data()
