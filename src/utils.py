import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

plt.rc('animation', html='html5')


def animate_scan(image):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(image[:, :, i].transpose()), cmap='viridis')
        return [im]

    return animation.FuncAnimation(fig, animate, frames=range(0, image.shape[2]), interval=200, blit=True)


def load_animate_scan(path):
    image = tsa.read_data(path)
    return animate_scan(image)
