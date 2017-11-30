"""Utility functions for TSA challenge."""
import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd

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
    return labels


def get_priors(zone_name='zone_num'):
    """Get baseline frequencies by threat zone."""
    labels = get_labels()
    return labels.groupby(zone_name).Probability.mean()
