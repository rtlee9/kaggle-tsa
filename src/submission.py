"""Generate submission from fitted models."""

import argparse
from os import path
from tqdm import tqdm

import torch
from torch.autograd import Variable

from .model import TsaNet
from .utils import get_labels, get_priors, generate_submission
from .pipeline import get_data_loaders
from .config import path_model


def main(submission_name):
    """Generate predictions and persist to disk."""
    # load data from disk
    zone_probs = get_priors()
    submissions = get_labels('submissions')
    submissions.set_index('Id', inplace=True)
    submissions['prediction'] = submissions.zone_num.map(lambda zone: zone_probs[zone])

    # read model list
    with open(path.join(path_model, 'model_list.txt'), 'r') as f:
        model_list = f.read().split('\n')
    model_list = [m for m in model_list if m != '']  # filter out blank lines

    for model_name in tqdm(model_list):

        model_filename = path.join(path_model, model_name)
        threat_zone = int(model_name.split('tz')[1].split('_')[0])

        # load model from disk
        model = TsaNet()
        model.load_state_dict(torch.load(model_filename))
        model.eval()
        model.cuda()

        loader_train, loader_validation, loader_submission = get_data_loaders(threat_zone)

        # collect predictions
        for data in loader_submission:
            images, target, mirrors, subject_idx, zone_nums = data['image'], data['threat'], data['mirror'], data['id'], data['zone']
            images, target, mirrors = images.cuda(), target.cuda(), mirrors.cuda()
            images, target, mirrors = Variable(images), Variable(target), Variable(mirrors)

            output = model(images, mirrors)
            for subject_id, prediction, zone_num in zip(subject_idx, output, zone_nums):
                id_ = '{}_Zone{}'.format(subject_id, zone_num)
                submissions.loc[id_, 'prediction'] = prediction.cpu().data[0]

    assert submissions.shape[0] == 1700

    # generate prediction and save
    generate_submission(submissions.prediction, submission_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSA challenge training')
    parser.add_argument('-S', '--submission-name', type=str, help='TSA model name')
    args = parser.parse_args()
    main(args.submission_name)
