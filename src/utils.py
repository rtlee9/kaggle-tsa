import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

plt.rc('animation', html='html5')


def animate_scan(path):
    data = tsa.read_data(path)
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
        return [im]

    return animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]), interval=200, blit=True)