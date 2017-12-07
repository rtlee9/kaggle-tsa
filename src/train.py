"""Train TSA Net and generate submission."""

import time
from os import path
import hashlib
import argparse
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from . import config, constants
from .pipeline import get_data_loaders
from .model import TsaNet
from .utils import get_run_details


def drop_learning_rate(optimizer, decay_factor=.4):
    """Decay the learning rate manually."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
        print('Learning rate decreased to {:.2e}'.format(param_group['lr']))


def hash_plaintext(plaintext):
    """Create hash representation of a plaintext string."""
    hash_object = hashlib.sha1(bytes(plaintext.encode('ascii')))
    return hash_object.hexdigest()


def main(threat_zone):
    """Train threat zone specific model."""
    loader_train, loader_validation, loader_submission = get_data_loaders(threat_zone)
    threat_ratio = loader_train.dataset.labels.Probability.value_counts(normalize=True)[1]
    threat_ratio_val = loader_validation.dataset.labels.Probability.value_counts(normalize=True)[1]

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

    # baseline_status
    validation_targets = torch.cat([l['threat'] for l in loader_validation])
    validation_targets = Variable(validation_targets)
    if config.verbose > 0:
        print('{:.1f}% of training labels are threat positive ({:.1f}% validation)'.format(threat_ratio * 100, threat_ratio_val * 100))
        print('Baseline guesses would yield {:.2f} BCE validation score'.format(
            F.binary_cross_entropy(Variable(torch.FloatTensor(len(validation_targets)).fill_(1) * threat_ratio), validation_targets.type(torch.FloatTensor)).data[0]
        ))

    # train model
    model.train()
    t0 = time.time()
    val_hist = []
    loss_hist = []
    for epoch in range(constants.N_EPOCHS):
        epoch_loss = []
        for batch_num, data in enumerate(tqdm(loader_train)):
            images, target = data['image'], data['threat']
            images, target = images.cuda(), target.cuda()
            images, target = Variable(images), Variable(target)

            for step_num in range(3):
                optimizer.zero_grad()
                output = model(images)
                loss = F.binary_cross_entropy(output, target.type(torch.cuda.FloatTensor))
                loss.backward()
                optimizer.step()

            epoch_loss.append(loss.data[0])

        # calculate validation accuracy
        model.eval()
        bce_values = torch.cuda.FloatTensor()
        for validation_data in loader_validation:
            # load validation data into GPU memory
            validation_images, validation_targets = validation_data['image'], validation_data['threat']
            validation_images, validation_targets = validation_images.cuda(), validation_targets.cuda()
            validation_images, validation_targets = Variable(validation_images), Variable(validation_targets)
            output_val = model(validation_images)
            output_val = (output_val * threat_ratio_val / threat_ratio).clamp(max=1)  # adjust validation output to account for threat ratio mismatch
            bce_values = torch.cat((bce_values, F.binary_cross_entropy(output_val, validation_targets.type(torch.cuda.FloatTensor)).data))
        bce_val = bce_values.mean()
        del validation_images, validation_targets, bce_values, output_val
        model.train()

        # decay learning rate if validation performance worsens
        if len(val_hist) > 0 and bce_val > val_hist[-1]:
            drop_learning_rate(optimizer)

        # append current validation performance to history
        val_hist.append(bce_val)
        loss_hist.append(sum(epoch_loss) / len(epoch_loss))

        print('Epoch {:2d} train / validation log loss [mean / min / max prediction]:\t{:.3f} / {:.3f}\t[{:.2f} / {:.2f} / {:.2f}]'.format(
            epoch,
            sum(epoch_loss) / len(epoch_loss),
            bce_val,
            output.mean().data[0],
            output.min().data[0],
            output.max().data[0],
        ))

    if config.verbose > 0:
        print('Training completed in {:.1f} minutes'.format((time.time() - t0) / 60))

    # save model state and description to disk
    run_details = get_run_details(
        model,
        optimizer,
        val_hist,
        loss_hist,
        specifications=dict(
            batch_size=constants.BATCH_SIZE,
            n_epochs=epoch + 1,
            train_test_split=constants.TRAIN_TEST_SPLIT_RATIO,
            image_dim=constants.IMAGE_DIM,
        ))
    model_hash = hash_plaintext(json.dumps(run_details))
    model_name = 'tz{tz:}_{v:.3f}_{n}'.format(
        tz=str(threat_zone).zfill(2),
        v=bce_val,
        n=model_hash,
    )
    torch.save(model.state_dict(), path.join(config.path_model, model_name))
    with open(path.join(config.path_model, model_hash + '.txt'), 'w') as f:
        f.write(str(model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSA challenge training')
    parser.add_argument('-Z', '--threat-zone', type=int, metavar='T', help='TSA threat zone')
    args = parser.parse_args()
    main(args.threat_zone)
