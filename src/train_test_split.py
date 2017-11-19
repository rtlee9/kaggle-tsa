import re
import os

from .constants import *


def get_train_test_file_list():

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        file_list = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER)
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(file_list) - max(int(len(file_list) * TRAIN_TEST_SPLIT_RATIO), 1)
        train_set = file_list[:train_test_split]
        test_set = file_list[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format(
              len(file_list) - train_test_split, len(file_list)))
    return file_list, train_set, test_set


if __name__ == '__main__':
    print(get_train_test_file_list())
