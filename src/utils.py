import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

plt.rc('animation', html='html5')


def _plot_image(image, title=None):
    plt.imshow(np.flipud(image))
    if title:
        plt.title(title)


def plot_image(*args, **kwargs):
    _plot_image(*args, **kwargs)
    plt.show()


def save_image(filename, *args, **kwargs):
    _plot_image(*args, **kwargs)
    plt.savefig(filename)


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