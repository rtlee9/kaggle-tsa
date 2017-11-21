import numpy as np
import pandas as pd
from os import path

from timeit import default_timer as timer
import tsahelper.tsahelper as tsa


from .constants import STAGE1_LABELS, INPUT_FOLDER, PREPROCESSED_DATA_FOLDER, BATCH_SIZE, EXAMPLES_PER_SUBJECT, verbose, BINARY_IMAGE_THRESHOLD

def preprocess_image(img, threat_zone, crop_dims):
    base_img = np.flipud(img)
    rescaled_img = tsa.convert_to_grayscale(base_img)
    high_contrast_img = tsa.spread_spectrum(rescaled_img)
    masked_img = tsa.roi(high_contrast_img, threat_zone)
    cropped_img = tsa.crop(masked_img, crop_dims)
    normalized_img = tsa.normalize(cropped_img)
    zero_centered_img = tsa.zero_center(normalized_img)
    return zero_centered_img


def preprocess_tsa_data():

    # OPTION 1: get a list of all subjects for which there are labels
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    #SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]

    # OPTION 3: get a list of subjects for small bore test purposes
    # SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
    #                 '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
    #                 '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']

    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()

    for subject in SUBJECT_LIST:

        # read in the images
        if verbose > 0:
            print('--------------------------------------------------------------')
            print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer() - start_time, subject))
            print('--------------------------------------------------------------')

        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, (threat_zone, crop_dims) in enumerate(zip(tsa.zone_slice_list, tsa.zone_crop_list)):

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num,
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                if verbose > 1:
                    print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                    print('Threat Zone Label -> {}'.format(label))

                if threat_zone[img_num] is not None:
                    preprocessed_img = preprocess_image(img, threat_zone[img_num], crop_dims[img_num])
                    threat_zone_examples.append([[tz_num], preprocessed_img, label])
                else:
                    if verbose > 1:
                        print('-> No view of tz:{} in img:{}. Skipping to next...'.format(tz_num, img_num))
                if verbose > 1:
                    print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact,
        # so this section just writes out the the data once there is a full minibatch
        # complete.
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + path.join(
                    PREPROCESSED_DATA_FOLDER,
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(
                        tz_num + 1,
                        len(threat_zone_examples[0][1][0]),
                        len(threat_zone_examples[0][1][1]),
                        batch_num)))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] ==
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]]
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a
                # tz_num 1 based in the minibatch file to select which batches to
                # use for training a given threat zone
                np.save(path.join(PREPROCESSED_DATA_FOLDER,
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(
                            tz_num + 1,
                            len(threat_zone_examples[0][1][0]),
                            len(threat_zone_examples[0][1][1]),
                            batch_num)),
                        tz_examples_to_save)
                del tz_examples_to_save

            # reset for next batch
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1

    # we may run out of subjects before we finish a batch, so we write out
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + path.join(PREPROCESSED_DATA_FOLDER,
                  'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(
                      tz_num + 1,
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]),
                      batch_num)))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] ==
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]]
                                        for features_label in tz_examples])

            # save batch
            np.save(path.join(PREPROCESSED_DATA_FOLDER,
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(
                        tz_num + 1,
                        len(threat_zone_examples[0][1][0]),
                        len(threat_zone_examples[0][1][1]),
                        batch_num)),
                    tz_examples_to_save)


def crop_image(image):
    """Find the edges of a TSA scan along each dimension and return the cropped image."""
    image_binary = (image > BINARY_IMAGE_THRESHOLD) * 1
    image_binary.sum()

    s0 = image_binary.mean(axis=1).mean(axis=1)
    s1 = image_binary.mean(axis=0).mean(axis=1)
    s2 = image_binary.mean(axis=0).mean(axis=0)

    m0 = s0 < .00015
    top_border = np.argmax(m0)
    if top_border == 0:
        top_border = s0.shape[0]
    top_border

    middle = np.floor(s1.shape[0] / 2)
    idx = np.arange(s1.shape[0])
    m1 = s1 < .0002
    right_border = np.argmax(np.where(idx > middle, m1, False))
    left_border = np.argmax(~np.where(idx < middle, m1, False))
    left_border, right_border

    middle = np.floor(s2.shape[0] / 2)
    idx = np.arange(s2.shape[0])
    m2 = s2 < .01
    back_border = np.argmax(np.where(idx > middle, m2, False))
    front_border = np.argmax(~np.where(idx < middle, m2, False))
    front_border, back_border

    return image[:top_border, left_border:right_border, front_border:back_border]


if __name__ == '__main__':
    preprocess_tsa_data()
