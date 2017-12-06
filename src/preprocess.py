"""Preprocess raw a3d scan images for use in model training."""
import numpy as np
from skimage.transform import resize
from os import path
import json
import tsahelper.tsahelper as tsa
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .config import path_a3d, path_cache, verbose, path_logs
from .constants import IMAGE_DIM
from .utils import get_labels


class A3DScans(Dataset):
    """A3d scans data."""

    def __init__(self, labels=None, keep_label_idx=None, blacklist=None, transforms=None):
        """Initialize dataset with optional filters."""
        if labels is None:
            labels = get_labels()
        if keep_label_idx is not None:
            labels = labels[labels.subject_id.isin(keep_label_idx)]
        if blacklist is not None:
            labels = labels[~labels.subject_id.isin(blacklist)]

        # map to common zone (across left-right center) and filter
        self.subject_ids = labels.subject_id.unique()
        self.transforms = transforms

    def __len__(self):
        """Get length of dataset."""
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """Get data element at index `idx`."""
        # parse idx
        subject_id = self.subject_ids[idx]
        image = tsa.read_data(path.join(path_a3d, subject_id + '.a3d'))
        # image = np.load(path.join(path_cache, subject_id + '.npy'))

        if self.transforms:
            image = self.transforms(image)

        return dict(image=image, subject_id=subject_id)


def moving_average(t, n):
    """Return the moving average series of a with kernel size n."""
    ma = nn.AvgPool1d(n - 1, stride=1, padding=4)
    return ma(t.unsqueeze(1)).squeeze()


def derivative(a, n):
    """Return the first derivative of series a with kernel size n."""
    return a - np.roll(a, n)


def find_edges(a, buffer=0, plot_distr=False):
    """Find the edges of a series."""
    ma = moving_average(a, 10)
    f = ma > 10000

    _, lower = torch.max(f, 1)
    _, upper = torch.max(reverse_tensor(f), 1)

    lower = lower.clamp(min=0, max=f.size(1))
    upper = f.size(1) - upper.clamp(min=0, max=f.size(1)).type(torch.cuda.FloatTensor)
    return lower, upper


def reverse_tensor(t):
    """Reverse tensor t."""
    dim = len(t.size()) - 1
    tv = Variable(torch.cuda.LongTensor(range(t.size(dim) - 1, -1, -1)))
    return t.index_select(dim, tv)


def crop_image(image, buffer=0):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    timg = rescale(Variable(image))
    avg_pool = nn.AvgPool3d(2, 1, )
    convolved = avg_pool(timg) * 2 ** 3  # convert to sum pool
    convolved = convolved.squeeze()
    filtered = (convolved * (convolved > 250).type(torch.cuda.FloatTensor))

    s0 = filtered.sum(dim=2).sum(dim=2)
    s1 = filtered.sum(dim=1).sum(dim=2)
    s2 = filtered.sum(dim=1).sum(dim=1)

    # borders for each dimension
    bottom, top = find_edges(s0, buffer)
    left, right = find_edges(s1, buffer)
    front, back = find_edges(s2, buffer)

    return [
        image[i, :int(t.data[0]), int(l.data[0]):int(r.data[0]), int(f.data[0]):int(b.data[0])]
        for i, (t, l, r, f, b) in enumerate(zip(top, left, right, front, back))
    ]


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
    a3d_scans = A3DScans(labels=scans)
    loader = DataLoader(
        a3d_scans,
        num_workers=4,
        batch_size=2,
        shuffle=True,
    )
    for batch in tqdm(loader):
        cropped_images = crop_image(batch['image'].cuda())
        for cropped_image, subject_id in zip(cropped_images, batch['subject_id']):
            image = cropped_image.cpu().numpy()
            resized_image = resize(image, (IMAGE_DIM, IMAGE_DIM, IMAGE_DIM), mode='constant')
            np.save(path.join(path_cache, subject_id + '.npy'), resized_image)
            crop_log[subject_id] = image.shape

    with open(path.join(path_logs, 'crop_log_{}.json'.format(type)), 'w') as f:
        json.dump(crop_log, f, indent=4)


if __name__ == '__main__':
    preprocess_tsa_data()
    preprocess_tsa_data('submissions')
