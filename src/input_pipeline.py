import numpy as np
import os

from .train_test_split import get_train_test_file_list
from .constants import PREPROCESSED_DATA_FOLDER


def input_pipeline(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []

    # Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))

    # Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)

    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])

    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)

    return feature_batch, label_batch

if __name__ == '__main__':
    file_list, train_set_file_list, test_set_file_list = get_train_test_file_list()
    print ('Train Set -----------------------------')
    for f_in in train_set_file_list:
        feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
        print (' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                    len(feature_batch[0]),
                                                    len(feature_batch[0][0])))
        print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))

    print ('Test Set -----------------------------')
    for f_in in test_set_file_list:
        feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
        print (' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                    len(feature_batch[0]),
                                                    len(feature_batch[0][0])))
        print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
