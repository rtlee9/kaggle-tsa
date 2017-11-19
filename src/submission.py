from os import path
import pandas as pd
import numpy as np
import tsahelper.tsahelper as tsa

from .model import alexnet
from .constants import INPUT_FOLDER, IMAGE_DIM, MODEL_DESCRIPTION, LEARNING_RATE
from . import config


def preprocess_subject_image_tz(subject, tz_num):
    images = tsa.read_data(path.join(INPUT_FOLDER, subject + '.aps'))
    images = images.transpose()
    tz_idx = tz_num - 1
    threat_zone = tsa.zone_slice_list[tz_idx]
    crop_dims = tsa.zone_crop_list[tz_idx]
    return [preprocess_image(img, threat_zone[i], crop_dims[i]) for i, img in enumerate(images) if threat_zone[i] is not None]


def preprocess_image(img, threat_zone, crop_dims):
    base_img = np.flipud(img)
    rescaled_img = tsa.convert_to_grayscale(base_img)
    high_contrast_img = tsa.spread_spectrum(rescaled_img)
    masked_img = high_contrast_img
    cropped_img = tsa.crop(masked_img, crop_dims)
    normalized_img = tsa.normalize(cropped_img)
    zero_centered_img = tsa.zero_center(normalized_img)
    return zero_centered_img


def get_pred(model, subject_id, zone_num):
    imgs = preprocess_subject_image_tz(subject_id, zone_num)
    preds = model.predict(np.array(imgs).reshape(-1, IMAGE_DIM, IMAGE_DIM, 1))
    return preds[:, 1].mean()


def generate_submissions():

    # load submissions data from disk
    submissions_df = pd.read_csv(path.join('data', 'stage1_sample_submission.csv'))
    submissions_df['subject_id'] = submissions_df.Id.str.split('_').str[0]
    submissions_df['zone_num'] = submissions_df.Id.str.split('Zone').str[1].astype(int)

    preds = {}
    for threat_zone in submissions_df.zone_num.unique():
        df = submissions_df[submissions_df.zone_num == threat_zone]

        # load model from disk
        model_name = '{}-tz-{}'.format(MODEL_DESCRIPTION, threat_zone)
        model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
        model.load(path.join(config.path_model, model_name + '.pk'))

        # sanity check
        sample_id = 0
        subject = df.iloc[sample_id]
        sample_subject_id = subject.subject_id
        sample_zone_num = subject.zone_num
        sample_pred = get_pred(model, sample_subject_id, sample_zone_num)
        assert sample_pred > 0
        assert sample_pred < 1

        # generate prediction
        preds[df.Id] = get_pred(model, df.subject_id, threat_zone)

    # collect predictions and write to disk
    predictions_df = pd.Series(preds, name='Probability')
    predictions_df.to_csv(path.join(config.path_submissions, MODEL_DESCRIPTION + '.csv'), header=True, index_label='Id')


if __name__ == '__main__':
    generate_submissions()