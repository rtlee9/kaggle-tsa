"""Train TSA Net and generate submission."""

import time
from os import path
import hashlib
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from . import config, constants
from .pipeline import get_data_loaders
from .model import TsaNet
from .utils import get_labels
from .constants import PREDICTION_NORM


def hash_model(model):
    """Create hash representation of a model based on its __repr__."""
    hash_object = hashlib.sha1(bytes(str(model).encode('ascii')))
    return hash_object.hexdigest()


def main(threat_zone):
    """Train threat zone specific model."""
    loader_train, loader_validation, loader_submission = get_data_loaders(threat_zone)
    labels = get_labels()
    tz_labels = labels[labels.zone_num == threat_zone]
    threat_ratio = tz_labels.Probability.value_counts(normalize=True)
    norm_factor = PREDICTION_NORM * threat_ratio[0]

    model = TsaNet()
    model.cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=constants.LR,
        momentum=constants.MOMENTUM,
        dampening=constants.DAMPENING,
        weight_decay=constants.L2_PENALTY,
    )

    # load validation data
    validation_data_loaded = [l for l in loader_validation]
    assert len(validation_data_loaded) == 1
    validation_data = validation_data_loaded[0]
    assert len(validation_data) == 3

    # load validation data into GPU memory
    validation_images, validation_targets = validation_data['image'], validation_data['threat']
    validation_images, validation_targets = validation_images.cuda(), validation_targets.cuda()
    validation_images, validation_targets = Variable(validation_images), Variable(validation_targets)

    # baseline stats
    if config.verbose > 0:
        print('{:.1f}% of labels are threat positive'.format(threat_ratio[1] * 100))
        print('Baseline guesses would yield {:.2f} BCE score'.format(
            F.binary_cross_entropy(Variable(torch.cuda.FloatTensor(len(validation_targets)).fill_(1) * threat_ratio[1]), validation_targets.type(torch.cuda.FloatTensor)).data[0]
        ))

    # train model
    model.train()
    t0 = time.time()
    for epoch in range(constants.N_EPOCHS):
        for data in tqdm(loader_train):
            images, target = data['image'], data['threat']
            images, target = images.cuda(), target.cuda()
            images, target = Variable(images), Variable(target)
            optimizer.zero_grad()
            output = model(images)
            loss = F.binary_cross_entropy_with_logits(output, target.type(torch.cuda.FloatTensor))
            loss.backward()
            optimizer.step()

        # print validation accuracy
        output_val = torch.sigmoid(model(validation_images)) / norm_factor
        print('Epoch {} train / validation log loss: {:.6f} / {:.6f}'.format(
            epoch,
            F.binary_cross_entropy(torch.sigmoid(output) / norm_factor, target.type(torch.cuda.FloatTensor)).data[0],
            F.binary_cross_entropy(output_val, validation_targets.type(torch.cuda.FloatTensor)).data[0],
        ))
        print('Epoch {} train / validation MAE loss: {:.6f} / {:.6f}'.format(
            epoch,
            F.l1_loss(torch.sigmoid(output) / norm_factor, target.type(torch.cuda.FloatTensor)).data[0],
            F.l1_loss(output_val, validation_targets.type(torch.cuda.FloatTensor)).data[0],
        ))

    if config.verbose > 0:
        print('Training completed in {:.1f} minutes'.format((time.time() - t0) / 60))

    # save model state and description to disk
    model_name = 'TSA_net_{n}_opt_{opt}_epochs_{e}_lr_{lr}_momentum_{m}_l2_{l2}'.format(
        n=hash_model(model),
        e=epoch + 1,
        opt=str(type(optimizer)).split('.')[-1].split("'")[0],
        lr=constants.LR,
        m=constants.MOMENTUM,
        l2=constants.L2_PENALTY,
    )
    torch.save(model.state_dict(), path.join(config.path_model, model_name))
    with open(path.join(config.path_model, hash_model(model) + '.txt'), 'w') as f:
        f.write(str(model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSA challenge training')
    parser.add_argument('-Z', '--threat-zone', type=int, metavar='T', help='TSA threat zone')
    args = parser.parse_args()
    main(args.threat_zone)
