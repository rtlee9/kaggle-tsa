"""Configuration variables for TSA challenge."""
from os import path, getenv


# define paths
path_src = path.dirname(path.abspath(__file__))
path_root = path.dirname(path_src)
path_data = path.join(path_root, 'data')
path_model = path.join(path_root, 'model')
path_plots = path.join(path_root, 'plots')
path_cache = path.join(path_root, 'cache')
path_submissions = path.join(path_root, 'submissions')
path_logs = path.join(path_root, 'logs')

path_sample_submissions = path.join(path_data, 'stage1_sample_submission.csv')
path_sample_submissions2 = path.join(path_data, 'stage2_sample_submission.csv')
path_aps = path.join(path_data, 'aps')
path_labels = path.join(path_data, 'stage1_labels.csv')

path_external_storage = getenv('EXTERNAL_STORAGE')
path_a3d = path.join(path_external_storage, 'a3d')

# define run settings
verbose = 1
stage = int(getenv('STAGE'))
