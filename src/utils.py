"""Utility functions for TSA challenge."""
import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from .config import path_labels, path_sample_submissions
from .zones import left_right_map

plt.rc('animation', html='html5')


def _plot_image(image, title=None):
    plt.imshow(np.flipud(image))
    if title:
        plt.title(title)


def plot_image(*args, **kwargs):
    """Plot image to matplotlib.pyplot."""
    _plot_image(*args, **kwargs)
    plt.show()


def save_image(filename, *args, **kwargs):
    """Save image to disk as png file."""
    _plot_image(*args, **kwargs)
    plt.savefig(filename)


def animate_scan(image, fig_size=(8, 8)):
    """Animate a 3 dimensional image by lapsing across the third dimension."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(image[:, :, i]), cmap='viridis')
        return [im]

    return animation.FuncAnimation(fig, animate, frames=range(0, image.shape[2]), interval=200, blit=True)


def load_animate_scan(path, *args, **kwargs):
    """Load a 3D image from from disk and animate."""
    image = tsa.read_data(path)
    return animate_scan(image, *args, **kwargs)


def get_labels(type='labels'):
    """Read labels / submissions from disk, parse and return."""
    if type == 'submissions':
        labels = pd.read_csv(path_sample_submissions)
    else:
        labels = pd.read_csv(path_labels)
    labels['subject_id'] = labels.Id.str.split('_').str[0]
    labels['zone_num'] = labels.Id.str.split('Zone').str[1].astype(int)
    labels['common_zone'] = labels.zone_num.map(
        lambda zone: left_right_map.get(zone, zone))

    # override incorrect labels https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/37615
    labels.loc[labels.Id == '623c761b4db398ea2157e6c5cd6c8c58_Zone3', 'Probability'] = 0
    labels.loc[labels.Id == '623c761b4db398ea2157e6c5cd6c8c58_Zone5', 'Probability'] = 1
    labels.loc[labels.Id == '623c761b4db398ea2157e6c5cd6c8c58_Zone11', 'Probability'] = 1
    labels.loc[labels.Id == '623c761b4db398ea2157e6c5cd6c8c58_Zone16', 'Probability'] = 1
    labels.loc[labels.Id == '496ec724cc1f2886aac5840cf890988a_Zone3', 'Probability'] = 1
    labels.loc[labels.Id == '496ec724cc1f2886aac5840cf890988a_Zone5', 'Probability'] = 0
    labels.loc[labels.Id == '496ec724cc1f2886aac5840cf890988a_Zone11', 'Probability'] = 0
    labels.loc[labels.Id == '496ec724cc1f2886aac5840cf890988a_Zone16', 'Probability'] = 0

    return labels


def get_priors(zone_name='zone_num'):
    """Get baseline frequencies by threat zone."""
    labels = get_labels()
    return labels.groupby(zone_name).Probability.mean()


def generate_submission(predictions, model_name):
    """Generate submission CSV file from predictions Pandas series."""
    predictions.name = 'Probability'
    predictions.to_csv('submissions/{}.csv'.format(model_name), header=True, float_format='%.7f')


def get_random_subject_id(labels):
    """Draw a random subject id from the labels dataset."""
    return labels.sample(1).iloc[0].subject_id


def plot_crop_boundaries(image, dims):
    """Plot an image and overlay it with a box representing crop dimensions."""
    width0 = dims[0][1] - dims[0][0]
    width1 = dims[1][1] - dims[1][0]

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(np.flipud(image))

    # Create a rectangle patch and add to axis
    rect = patches.Rectangle((dims[1][0], image.shape[0] - dims[0][1]), width1, width0, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()


def plot_line(np_array):
    """Plot a numpy array as a line."""
    pd.Series(np_array).plot.line()
    plt.show()


def moving_average(a, n):
    """Return the moving average series of a with kernel size n."""
    return convolve(a, np.ones(n))


def derivative(a, n):
    """Return the first derivative of series a with kernel size n."""
    return a - np.roll(a, n)


def get_model_structure(model):
    """Return JSON serialized model structure."""
    # TODO: recurse model tree -- don't assume two layers
    return [[str(c) for c in p] for p in model.named_children()]


def get_component_details(model, structure_parser):
    """Return JOSN representation of a PyTorch model."""
    return dict(
        type=type(model).__name__,
        structure=structure_parser(model),
    )


def get_hyperparameters(optimizer):
    """Get hyperparameters from PyTorch object."""
    desc_keys = [k for k in optimizer.param_groups[0].keys() if k != 'params']
    desc = [{k: p[k] for k in desc_keys} for p in optimizer.param_groups]
    return desc


def get_run_details(model, optimizer, validation_loss=[], training_loss=[], specifications={}):
    """Get serializable training run details."""
    return dict(
        model=get_component_details(model, get_model_structure),
        optimizer=get_component_details(optimizer, get_hyperparameters),
        validation_loss=validation_loss,
        training_loss=training_loss,
        **specifications
    )
