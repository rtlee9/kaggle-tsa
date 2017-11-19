from os import path
import pandas as pd
import numpy as np
import tsahelper.tsahelper as tsa

from .model import alexnet
from .constants import INPUT_FOLDER, IMAGE_DIM, MODEL_NAME, LEARNING_RATE
from . import config


def preprocess_subject_image_tz(subject, tz_num):
    images = tsa.read_data(path.join(INPUT_FOLDER, subject + '.aps'))
    images = images.transpose()
    threat_zone = tsa.zone_slice_list[tz_num]
    crop_dims = tsa.zone_crop_list[tz_num]
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
    # load model from disk
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    model.load(path.join(config.path_model, MODEL_NAME + '.pk'))

    # load submissions data from disk
    submissions_df = pd.read_csv(path.join('data', 'stage1_sample_submission.csv'))
    submissions_df['subject_id'] = submissions_df.Id.str.split('_').str[0]
    submissions_df['zone_num'] = submissions_df.Id.str.split('Zone').str[1].astype(int)
    submissions_df['zone_idx'] = submissions_df.zone_num - 1

    # sanity check
    sample_id = 0
    subject = submissions_df.iloc[sample_id]
    sample_subject_id = subject.subject_id
    sample_zone_num = subject.zone_num
    sample_pred = get_pred(model, sample_subject_id, sample_zone_num)
    assert sample_pred > 0
    assert sample_pred < 1

    # generate prediction as mean across all images for a given subject
    submissions_df['Probability'] = submissions_df.apply(lambda row: get_pred(model, row['subject_id'], row['zone_idx']), axis=1)

    # write predictions to disk
    submissions_df[['Id', 'Probability']].to_csv(path.join(config.path_submissions, MODEL_NAME + '.csv'), index=False)


if __name__ == '__main__':
    generate_submissions()