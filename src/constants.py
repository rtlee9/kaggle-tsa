"""Constant variables for TSA challenge."""

# preprocessing constants
BATCH_SIZE = 16
TRAIN_TEST_SPLIT_RATIO = 0.1
IMAGE_DIM = 128
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
MODEL_DESCRIPTION = ('tsa-{}-lr-{}-{}-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM))
CROP_LOG_INTERVAL = 20

# training constants
BATCH_SIZE = 24
LOG_INTERVAL = 50
VALIDATION_INTERVAL = 200
LR = .01
MOMENTUM = .5
L2_PENALTY = .2e1
N_EPOCHS = 20
DAMPENING = .1
