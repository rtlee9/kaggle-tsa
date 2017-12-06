"""Pipeline for model training."""

from os import path
import json

import pandas as pd
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.transform import resize

from . import config
from .utils import get_labels
from .zones import center_zones, left_only_zones, left_right_map, common_threat_body_map
from .constants import BATCH_SIZE, TRAIN_TEST_SPLIT_RATIO
from .crop import hard_crop


def get_blacklist():
    """Generate blacklist from crop dimension logs."""
    with open(path.join(config.path_logs, 'crop_log_labels.json'), 'r') as f:
        crop_records = json.load(f)
    crop_records_df = pd.DataFrame.from_dict(crop_records, orient='index')
    crop_records_df['blacklist'] = np.where(
        (crop_records_df[0] < 500) | (crop_records_df[1] < 400) | (crop_records_df[2] < 100),
        1, 0
    )
    if config.verbose > 1:
        print('Blacklisting {:,} subjects due to bad cropping'.format(crop_records_df['blacklist'].value_counts(sort=False)[1]))
    return crop_records_df[crop_records_df.blacklist == 1].index.values


class TsaScansDataset(Dataset):
    """TSA scans dataset."""

    def __init__(self, threat_zone, labels=None, keep_label_idx=None, blacklist=None, transforms=None):
        """
        Args:
            labels (pd.DataFrame, optional): labels dataframe
            keep_label_idx (list[string], optional): List of labels to keep (used for train/test split).
            blacklist (list[string], optional): List of labels to blacklist
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if labels is None:
            labels = get_labels()
        if keep_label_idx is not None:
            labels = labels[labels.subject_id.isin(keep_label_idx)]
        if blacklist is not None:
            labels = labels[~labels.subject_id.isin(blacklist)]

        # map to common zone (across left-right center) and filter
        common_zone = left_right_map.get(threat_zone, threat_zone)
        labels = labels[labels.common_zone == common_zone]
        self.labels = labels.set_index('subject_id')
        self.transforms = transforms

    def __len__(self):
        """Get length of dataset."""
        return self.labels.shape[0]  # each subject gets one scan per l/r half

    def __getitem__(self, idx):
        """Get data element at index `idx`."""
        # parse idx
        data = self.labels.iloc[idx]
        subject_idx = data.name
        image = np.load(path.join(config.path_cache, subject_idx + '.npy'))

        if data.zone_num in left_only_zones:
            image = np.fliplr(image)

        if self.transforms:
            image = self.transforms(image)

        return dict(
            image=image,
            threat=data.Probability,
            id=subject_idx,
            zone=data.zone_num,
            common_zone=data.common_zone,
        )


class Resize(object):
    """Resize object to new size."""

    def __call__(self, images):
        """Downsample image for modeling."""
        return resize(images, (32, 32, 32), mode='constant')


class Filter(object):
    """Image filter transformer."""

    def __call__(self, image):
        """Filter low intensity pixels from image."""
        return image * (image > .05)


class ConditionalRandomFlip(object):
    """ConditionalRandomFlip transformer."""

    def __init__(self, threat_zone):
        """Initialize ConditionalRandomFlip with threat zone."""
        super().__init__()
        self.threat_zone = threat_zone

    def __call__(self, image):
        """Randomply flip an image horizontally."""
        if (self.threat_zone in center_zones) & (np.random.random() > .5):
            return np.fliplr(image)
        return image


class ZoneCrop(object):
    """Threat zone crop transformer."""

    def __init__(self, threat_zone):
        """Initialize ZoneCrop with threat zone."""
        super().__init__()
        threat_zone = threat_zone
        self.crop_dims = common_threat_body_map[threat_zone]

    def __call__(self, image):
        """Crop image to threat zone area."""
        return hard_crop(image, self.crop_dims)


class RandomRotation(object):
    """Random rotation transformation."""

    def __init__(self, range=(-1, 1), axes=(1, 2)):
        """Initialize RandomRotation with rotation parameters."""
        super().__init__()
        self.range = range
        self.axes = axes

    def __call__(self, image):
        """Rotate image by by a random degree around specified axes."""
        rotation = np.random.uniform(*self.range)
        return ndimage.rotate(image, rotation, self.axes, reshape=False, mode='reflect')


class RandomShear(object):
    """Random shear transformation."""

    def __init__(self, range=(-.02, .02)):
        """Initialize RandomShear with shear parameters."""
        super().__init__()
        self.range = range

    def __call__(self, image):
        """Shear image by a random amount along the x-axis within the range specified."""
        matrix = np.eye(3)
        shear_factor = np.random.uniform(*self.range)
        matrix[1, 0] = shear_factor
        return ndimage.affine_transform(image, matrix, mode='reflect')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        """Convert to float tensor and add two dimensions:
        one for the number of images and the other for initial chanels (no RGB, so this is 1).
        """
        return torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0)


class MeanVariance(object):
    """Remove mean and variance."""

    def __call__(self, image):
        """Rescale to range [-1, 1]."""
        return (image - image.min()) / (image.max() - image.min()) * 2 - 1


def get_data_loaders(threat_zone):
    """Get train, validation, and submission loaders."""
    # create train / validation split
    subject_idx = get_labels().subject_id.unique()
    label_idx_train, label_idx_validation = train_test_split(subject_idx, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=0)

    # create loader for training data
    blacklist = get_blacklist()
    train_transformations = [ZoneCrop(threat_zone), ConditionalRandomFlip(threat_zone), Resize(), RandomRotation(), RandomShear(), ToTensor(), MeanVariance()]  # training transformations
    test_transformations = [ZoneCrop(threat_zone), Resize(), ToTensor(), MeanVariance()]  # base transformations
    dataset_train = TsaScansDataset(
        threat_zone=threat_zone,
        keep_label_idx=label_idx_train,
        blacklist=blacklist,
        transforms=transforms.Compose(train_transformations)
    )
    loader_train = DataLoader(
        dataset_train,
        num_workers=4,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # create loader for validation data
    dataset_validation = TsaScansDataset(
        threat_zone=threat_zone,
        keep_label_idx=label_idx_validation,
        # blacklist=blacklist,
        transforms=transforms.Compose(test_transformations)
    )
    loader_validation = DataLoader(
        dataset_validation,
        batch_size=16,
        shuffle=False,
        num_workers=3,
    )

    # create loader for submission data
    dataset_submission = TsaScansDataset(
        threat_zone=threat_zone,
        labels=get_labels('submissions'),
        transforms=transforms.Compose(test_transformations),
    )
    loader_submission = DataLoader(
        dataset_submission,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
    )

    return loader_train, loader_validation, loader_submission
