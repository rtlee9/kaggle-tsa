"""Preprocess raw a3d scan images for use in model training."""
import numpy as np
from skimage.transform import resize
from os import path
import json
import tsahelper.tsahelper as tsa
from tqdm import tqdm
import torch
from torch import nn

from .config import path_a3d, path_cache, verbose, path_plots, path_logs
from .constants import IMAGE_DIM
from .utils import save_image, get_labels, moving_average


def find_edges(a, buffer=0, plot_distr=False):
    """Find the edges of a series."""
    ma = moving_average(a, 10)
    f = ma > 10000

    _, lower = torch.max(f, 0)
    _, upper = torch.max(reverse_tensor(f), 0)
    lower = lower.data[0]
    upper = f.size(0) - upper.data[0]
    if f.data[0]:
        lower = 0
    if f.data[-1]:
        upper = f.size(0)
    return max(lower - buffer, 0), min(upper + buffer, f.size(0))


def reverse_tensor(t):
    """Reverse tensor t."""
    dim = len(t.size()) - 1
    return t.index_select(dim, torch.cuda.LongTensor(range(t.size(dim) - 1, -1, -1)))


def crop_image(image, buffer=0):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    image = image.transpose(2, 0, 1)  # axis are now height (top) x width (side) x  depth (front)
    timg = rescale(torch.from_numpy(image).cuda())
    avg_pool = nn.AvgPool3d(2, 1, )
    convolved = avg_pool(timg.unsqueeze(0)) * 2 ** 3  # convert to sum pool
    convolved = convolved.squeeze()
    filtered = (convolved * (convolved > 250).type(torch.cuda.FloatTensor))

    s0 = filtered.sum(dim=1).sum(dim=1)
    s1 = filtered.sum(dim=0).sum(dim=1)
    s2 = filtered.sum(dim=0).sum(dim=0)

    # borders for each dimension
    bottom, top = find_edges(s0, buffer)
    left, right = find_edges(s1, buffer)
    front, back = find_edges(s2, buffer)

    resized_image = image[:top, left:right, front:back]
    if verbose > 1:
        print('Image resized from {} to {}'.format(image.shape, resized_image.shape))
    return resized_image


def rescale(img):
    """Rescale tensor image to range [0, 255]."""
    min_, max_ = img.min(), img.max()
    base_range = max_ - min_
    rescaled_range = 255 - 0
    return (img - min_) * rescaled_range / base_range


def preprocess_tsa_data(type='labels'):
    """Preprocess all a3d files for training and persist to disk."""
    scans = get_labels(type)
    crop_log = {}
    for subject_id in tqdm(scans.subject_id.unique()):
        image = tsa.read_data(path.join(path_a3d, subject_id + '.a3d'))
        cropped_image = crop_image(torch.from_numpy(image).cuda())
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
    preprocess_tsa_data()
    preprocess_tsa_data('submissions')
