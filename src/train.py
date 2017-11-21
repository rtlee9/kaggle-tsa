import argparse
from os import path
import random
import numpy as np

from .train_test_split import get_train_test_file_list
from .model import alexnet
from .input_pipeline import input_pipeline
from .constants import PREPROCESSED_DATA_FOLDER, IMAGE_DIM, LEARNING_RATE, N_TRAIN_STEPS, MODEL_DESCRIPTION, verbose
from . import config


def train_conv_net(threat_zone):

    model_name = '{}-tz-{}'.format(MODEL_DESCRIPTION, threat_zone)
    val_features = []
    val_labels = []

    # get train and test batches
    file_list, train_set_file_list, test_set_file_list = get_train_test_file_list(threat_zone)

    # instantiate model
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE, model_name)

    # read in the validation test set
    for j, test_f_in in enumerate(test_set_file_list):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in,
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    # start training process
    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
        random.shuffle(train_set_file_list)

        # run through every batch in the training set
        for f_in in train_set_file_list:

            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            if verbose > 1:
                print ('Feature Batch Shape ->', feature_batch.shape)

            # run the fit operation
            model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1,
                      validation_set=({'features': val_features}, {'labels': val_labels}),
                      shuffle=True, snapshot_step=None, show_metric=True,
                      run_id=model_name)

    # persist model to disk and remove from GPU memory
    model.save(path.join(config.path_model, model_name + '.pk'))
    del model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSA model training')
    parser.add_argument('-tz', '--threat-zone', type=int, help='TSA threat zone')
    args = parser.parse_args()
    train_conv_net(args.threat_zone)
