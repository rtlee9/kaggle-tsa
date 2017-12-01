"""Generate submission from fitted models."""

import argparse

import torch
from torch.autograd import Variable

from .model import TsaNet
from .utils import get_labels, get_priors, generate_submission
from .pipeline import get_data_loaders


def generate_baseline_preds(submissions):
    """Generate baseline frequency predictions based on zone zone number."""
    zone_probs = get_priors()
    zone_predictions = submissions.zone_num.map(lambda zone: zone_probs[zone])
    assert zone_predictions.nunique() == 10
    generate_submission(zone_predictions, 'baseline_predictions')


def generate_common_zone_baseline_preds(submissions):
    """Generate baseline frequency predictions based on common zone."""
    common_zone_probs = get_priors('common_zone')
    common_zone_predictions = submissions.common_zone.map(lambda zone: common_zone_probs[zone])
    assert common_zone_predictions.nunique() == 10
    generate_submission(common_zone_predictions, 'common_zone_predictions')


def main(threat_zone, model_name, _model_filename):
    """Generate predictions and persist to disk."""
    # load model from disk
    model = TsaNet()
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    model.cuda()

    # load data from disk
    zone_probs = get_priors()
    submissions = get_labels('submissions')
    submissions.set_index('Id', inplace=True)
    submissions['prediction'] = submissions.zone_num.map(lambda zone: zone_probs[zone])
    submissions['zeros'] = 0
    loader_train, loader_validation, loader_submission = get_data_loaders(threat_zone)

    # collect predictions
    for data in loader_submission:
        images, target, subject_idx, zone_nums = data['image'], data['threat'], data['id'], data['zone']
        images, target = images.cuda(), target.cuda()
        images, target = Variable(images), Variable(target)
        output = model(images)
        for subject_id, prediction, zone_num in zip(subject_idx, output, zone_nums):
            id_ = '{}_Zone{}'.format(subject_id, zone_num)
            submissions.loc[id_, 'prediction'] = prediction.cpu().data[0]
    assert submissions.shape[0] == 1700

    # print summary stats
    tz_predictions = submissions[submissions.common_zone == threat_zone].prediction
    print('Mean / min / max prediction: {:.3f} / {:.3f} / {:.3f}'.format(
        tz_predictions.mean(),
        tz_predictions.min(),
        tz_predictions.max(),
    ))

    # generate prediction and save
    generate_submission(submissions.prediction, model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSA challenge training')
    parser.add_argument('-Z', '--threat-zone', type=int, help='TSA threat zone')
    parser.add_argument('-M', '--model-name', type=str, help='TSA model name')
    args = parser.parse_args()
    model_filename = 'model/{}'.format(args.model_name)
    main(args.threat_zone, args.model_name, model_filename)
