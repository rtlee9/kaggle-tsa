import tsahelper.tsahelper as tsa
import matplotlib.pyplot as plt
import random
from matplotlib import animation
import numpy as np

from .constants import TRAIN_SET_FILE_LIST


def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list

plt.rc('animation', html='html5')


def plot_image(path):
    data = tsa.read_data(path)
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
        return [im]

    return animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]), interval=200, blit=True)

plot_image('data/aps/011516ab0eca7cad7f5257672ddde70e.aps')

if __name__ == '__main__':
    print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
    shuffle_train_set(TRAIN_SET_FILE_LIST)
    print ('After Shuffling ->', TRAIN_SET_FILE_LIST)
