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


def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the initial LR decayed by 10 every 30 epochs."""
    lr = constants.LR * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def hash_model(model):
    """Create hash representation of a model based on its __repr__."""
    hash_object = hashlib.sha1(bytes(str(model).encode('ascii')))
    return hash_object.hexdigest()


def main(threat_zone):
    """Train threat zone specific model."""
    loader_train, loader_validation, loader_submission = get_data_loaders(threat_zone)
    labels = loader_train.dataset.labels
    threat_ratio = labels.Probability.value_counts(normalize=True)

    model = TsaNet()
    model.cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=constants.LR,
        momentum=constants.MOMENTUM,
        nesterov=True,
        # dampening=constants.DAMPENING,
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
        print('{:.1f}% of training labels are threat positive'.format(threat_ratio[1] * 100))
        print('Baseline guesses would yield {:.2f} BCE score'.format(
            F.binary_cross_entropy(Variable(torch.cuda.FloatTensor(len(validation_targets)).fill_(1) * threat_ratio[1]), validation_targets.type(torch.cuda.FloatTensor)).data[0]
        ))

    # train model
    model.train()
    t0 = time.time()
    for epoch in range(constants.N_EPOCHS):
        adjust_learning_rate(optimizer, epoch)
        epoch_loss = []
        for data in loader_train:
            images, target = data['image'], data['threat']
            images, target = images.cuda(), target.cuda()
            images, target = Variable(images), Variable(target)
            optimizer.zero_grad()
            output = model(images)
            loss = F.binary_cross_entropy(output, target.type(torch.cuda.FloatTensor))
            epoch_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()

        # print validation accuracy
        output_val = model(validation_images)
        print('Epoch {} train / validation log loss [mean / min / max prediction]:\t{:.3f} / {:.3f}\t[{:.2f} / {:.2f} / {:.2f}]'.format(
            epoch,
            sum(epoch_loss) / len(epoch_loss),
            F.binary_cross_entropy(output_val, validation_targets.type(torch.cuda.FloatTensor)).data[0],
            output.mean().data[0],
            output.min().data[0],
            output.max().data[0],
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
