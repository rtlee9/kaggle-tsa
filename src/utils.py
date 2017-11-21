import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from .constants import BINARY_IMAGE_THRESHOLD

plt.rc('animation', html='html5')


def plot_image(image, title=None):
    plt.imshow(np.flipud(image))
    if title:
        plt.title(title)
    plt.show()


def animate_scan(image, fig_size=(8, 8)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(image[:, :, i]), cmap='viridis')
        return [im]

    return animation.FuncAnimation(fig, animate, frames=range(0, image.shape[2]), interval=200, blit=True)


def load_animate_scan(path, *args, **kwargs):
    image = tsa.read_data(path)
    return animate_scan(image, *args, **kwargs)


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
