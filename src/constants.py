from os import path
from . import config


INPUT_FOLDER = config.path_aps
PREPROCESSED_DATA_FOLDER = config.path_cache
STAGE1_LABELS = path.join(config.path_data, 'stage1_labels.csv')
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = path.join('logs', 'train/')
MODEL_PATH = path.join('logs', 'model/')
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE))

verbose = 1
