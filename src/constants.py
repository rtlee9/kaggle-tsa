"""Constant variables for TSA challenge."""
BATCH_SIZE = 16
TRAIN_TEST_SPLIT_RATIO = 0.1
IMAGE_DIM = 128
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
MODEL_DESCRIPTION = ('tsa-{}-lr-{}-{}-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM))
