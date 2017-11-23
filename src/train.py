"""Train TSA Net and generate submission."""

import time
from os import path
import hashlib
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from . import config, constants
from .pipeline import get_data_loaders
from .model import TsaNet
from .zones import left_zones, right_zones
from .utils import get_labels


def hash_model(model):
    """Create hash representation of a model based on its __repr__."""
    hash_object = hashlib.sha1(bytes(str(model).encode('ascii')))
    return hash_object.hexdigest()


def merge_lr_predictions(lr_predictions):
    """Merge predictions for the left and right sides of the body and map to full body space."""
    # map left predictions to full body space
    predictions_left = np.zeros(17)
    predictions_left[np.array(left_zones) - 1] = lr_predictions['left']

    # map right predictions to full body space
    predictions_right = np.zeros(17)
    predictions_right[np.array(right_zones) - 1] = lr_predictions['right']

    # merge predictions
    return np.maximum(predictions_left, predictions_right)


def test_merge_lr_predictions():
    """Test left right merge."""
    assert np.absolute(merge_lr_predictions({
        'left': np.array([0.50157225, 0.07031653, 0.05267072, 0.04668518, 0.05428256, 0.05049477, 0.06152101, 0.04983074, 0.05413319, 0.05849301]),
        'right': np.array([0.50156403, 0.07031745, 0.05267141, 0.04668608, 0.05428406, 0.05049552, 0.06152144, 0.0498316, 0.05413451, 0.05849396])
    }) - np.array([
        0.50157225, 0.07031653, 0.50156403, 0.07031745, 0.0498316,
        0.05267072, 0.05267141, 0.04668518, 0.05413451, 0.04668608,
        0.05428256, 0.05428406, 0.05049477, 0.05049552, 0.06152101,
        0.06152144, 0.05849396])).sum() < 1e-7

loader_train, loader_validation, loader_submission = get_data_loaders()
if config.verbose > 0:
    print('{:.1f}% of labels are threat positive'.format(get_labels().Probability.value_counts(normalize=True)[1] * 100))

model = TsaNet(num_classes=len(right_zones))
model.cuda()
optimizer = optim.SGD(
    model.parameters(),
    lr=constants.LR,
    momentum=constants.MOMENTUM,
    weight_decay=constants.L2_PENALTY,
)

# load validation data
validation_data_loaded = [l for l in loader_validation]
assert len(validation_data_loaded) == 1
validation_data = validation_data_loaded[0]
assert len(validation_data) == 4

# load validation data into GPU memory
validation_images, validation_targets = validation_data['image'], validation_data['threat']
validation_images, validation_targets = validation_images.cuda(), validation_targets.cuda()
validation_images, validation_targets = Variable(validation_images), Variable(validation_targets)
validation_targets = validation_targets.type(torch.cuda.FloatTensor)

# train model
model.train()
t0 = time.time()
for epoch in range(constants.N_EPOCHS):
    for batch_idx, data in enumerate(loader_train):
        images, target = data['image'], data['threat']
        images, target = images.cuda(), target.cuda()
        images, target = Variable(images), Variable(target)
        optimizer.zero_grad()
        output = model(images)
#         loss = F.multilabel_margin_loss(output, target.type(torch.cuda.LongTensor))
        loss = F.multilabel_soft_margin_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{:04d}/{} ({:02d}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(target),
                len(loader_train.dataset),
                int(100. * batch_idx / len(loader_train)),
                loss.data[0],
            ))

    # print validation accuracy
    validation_output = model(validation_images)
    validation_loss = F.mse_loss(validation_output, validation_targets)
    print('Validation MSE loss: {:.6f}'.format(validation_loss.data[0]))
    print('Validation MAE loss: {:.6f}'.format(torch.mean(torch.abs(validation_targets - validation_output)).data[0]))

if config.verbose > 0:
    print('Training completed in {:.1f} minutes'.format((time.time() - t0) / 60))

# save model state and description to disk
model_name = 'TSA_net_{n}_opt_{opt}_epochs_{e}_lr_{lr}_momentum_{m}_l2_{l2}'.format(
    n=hash_model(model),
    e=epoch,
    opt=str(type(optimizer)).split('.')[-1].split("'")[0],
    lr=constants.LR,
    m=constants.MOMENTUM,
    l2=constants.L2_PENALTY,
)
torch.save(model.state_dict(), path.join(config.path_model, model_name))
with open(path.join(config.path_model, hash_model(model) + '.txt'), 'w') as f:
    f.write(str(model))

# collect raw predictions from model
predictions_raw = {}
model.eval()
for batch_idx, data in enumerate(loader_submission):
    images, target, idx, left_indicator = data['image'], data['threat'], data['id'], data['left_indicator']
    images, target = images.cuda(), target.cuda()
    images, target = Variable(images), Variable(target)
    output = model(images)
    for id_, pred, left in zip(idx, output.data, left_indicator):
        results = predictions_raw.setdefault(id_, {})
        results['left' if left else 'right'] = pred.cpu().numpy()
        predictions_raw[id_] = results

# merge predictions and write to disk
predictions = {idx: merge_lr_predictions(lr_predictions) for idx, lr_predictions in predictions_raw.items()}
submissions_df = get_labels('submissions')
predictions_df = submissions_df.set_index('Id').apply(
    lambda row: predictions[row['subject_id']][row['zone_num'] - 1], axis=1)
predictions_df.name = 'Probability'
predictions_df.to_csv(
    path.join(config.path_root, 'submissions', 'test.csv'),
    header=True,
    index_label='Id',
    float_format='%.3f',
    encoding='utf-8'
)
