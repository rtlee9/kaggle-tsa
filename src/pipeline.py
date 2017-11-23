"""Pipeline for model training."""

from os import path
import json

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.transform import resize

from . import config
from .utils import get_labels
from .zones import left_zones, right_zones, left_right_map
from .constants import BATCH_SIZE


def get_blacklist():
    """Generate blacklist from crop dimension logs."""
    with open(path.join(config.path_logs, 'crop_log_labels.json'), 'r') as f:
        crop_records = json.load(f)
    crop_records_df = pd.DataFrame.from_dict(crop_records, orient='index')
    crop_records_df['blacklist'] = np.where(
        (crop_records_df[0] < 500) | (crop_records_df[1] < 400) | (crop_records_df[2] < 100),
        1, 0
    )
    if config.verbose > 0:
        print('Blacklisting {:,} subjects due to bad cropping'.format(crop_records_df['blacklist'].value_counts(sort=False)[1]))
    return crop_records_df[crop_records_df.blacklist == 1].index.values


class TsaScansDataset(Dataset):
    """TSA scans dataset."""

    def __init__(self, labels=None, keep_label_idx=None, blacklist=None, transforms=None):
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

        # map to common zone (across left-right center)
        labels['left'] = labels.zone_num.isin(left_zones)
        labels['right'] = labels.zone_num.isin(right_zones)
        labels['common_zone'] = labels.zone_num.map(
            lambda zone: left_right_map.get(zone, zone))
        labels.sort_values('common_zone', inplace=True)  # sort to ensure threat vector is constant across examples
        labels.set_index('subject_id', inplace=True)
        self.left_labels = labels[labels.left]
        self.right_labels = labels[labels.right]
        self.subject_ids = labels.index.unique()
        self.transforms = transforms

    def __len__(self):
        """Get length of dataset."""
        return self.subject_ids.shape[0] * 2  # each subject gets one scan per l/r half

    def __getitem__(self, idx):
        """Get data element at index `idx`."""
        # parse idx
        subject_idx = np.floor(idx / 2).astype(int)
        left_indicator = idx % 2 == 0  # True if left side, False if right side

        # get data
        subject_id = self.subject_ids[subject_idx]
        image = np.load(path.join(config.path_cache, subject_id + '.npy'))
        mid_point = np.floor(image.shape[1] / 2).astype(int)
        if left_indicator:
            image = np.fliplr(image[:, :mid_point + 5, :])
            threat = self.left_labels.loc[subject_id].Probability.values
        else:
            image = image[:, mid_point - 5:, :]
            threat = self.right_labels.loc[subject_id].Probability.values
        threat_tensor = torch.from_numpy(threat).type(torch.FloatTensor)
        assert threat.shape[0] == len(right_zones)

        if self.transforms:
            image = self.transforms(image)

        return dict(
            image=image,
            threat=threat_tensor,
            id=subject_id,
            left_indicator=left_indicator,
        )


class Resize(object):
    """Resize object to new size."""

    def __call__(self, images):
        """Downsample image for modeling."""
        return resize(images, (64, 32, 32), mode='constant')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        """Convert to float tensor and add two dimensions:
        one for the number of images and the other for initial chanels (no RGB, so this is 1).
        """
        return torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0) * 2


def get_data_loaders():
    """Get train, validation, and submission loaders."""
    # create train / validation split
    subject_idx = get_labels().subject_id.unique()
    label_idx_train, label_idx_validation = train_test_split(subject_idx, test_size=.01, random_state=0)
    label_idx_train.shape, label_idx_validation.shape

    # create loader for training data
    blacklist = get_blacklist()
    dataset_train = TsaScansDataset(
        keep_label_idx=label_idx_train,
        blacklist=blacklist,
        transforms=transforms.Compose([Resize(), ToTensor()])
    )
    assert type(dataset_train.__getitem__(0)['image']) == torch.FloatTensor
    assert (len(dataset_train) == 2206)
    assert (dataset_train.__getitem__(0))
    assert (dataset_train.__getitem__(len(dataset_train) - 1))
    assert np.all(dataset_train.left_labels.loc['00360f79fd6e02781457eda48f85da90'].Probability.values == np.zeros(len(right_zones)))
    assert np.all(dataset_train.right_labels.loc['00360f79fd6e02781457eda48f85da90'].Probability.values == np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    assert np.all(dataset_train.left_labels.loc['c893b3c645841e008cccb20c3b6c75df'].Probability.values == np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert np.all(dataset_train.right_labels.loc['c893b3c645841e008cccb20c3b6c75df'].Probability.values == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3,
    )

    # create loader for validation data
    dataset_validation = TsaScansDataset(
        keep_label_idx=label_idx_validation,
        # blacklist=blacklist,
        transforms=transforms.Compose([Resize(), ToTensor()])
    )
    loader_validation = DataLoader(
        dataset_validation,
        batch_size=len(label_idx_validation) * 2,
        shuffle=True,
        num_workers=3,
    )

    # create loader for submission data
    dataset_submission = TsaScansDataset(
        labels=get_labels('submissions'),
        transforms=transforms.Compose([Resize(), ToTensor()])
    )
    loader_submission = DataLoader(
        dataset_submission,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
    )
    assert len(loader_submission.dataset) == 100 * 2
    assert len(loader_submission) == np.ceil(100 * 2 / BATCH_SIZE).astype(int)

    return loader_train, loader_validation, loader_submission
