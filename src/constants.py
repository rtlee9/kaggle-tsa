"""Constant variables for TSA challenge."""

# preprocessing constants
TRAIN_TEST_SPLIT_RATIO = 0.03
IMAGE_DIM = 128
CROP_LOG_INTERVAL = 20

# training constants
BATCH_SIZE = 40
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 200
LR = 1e-4
MOMENTUM = .9
DAMPENING = .5
L2_PENALTY = 10e-4
N_EPOCHS = 10
N_WORKERS = 6
