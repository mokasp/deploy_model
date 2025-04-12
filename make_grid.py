#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from output_image import output_image
import numpy as np

def make_grid(rgb_values):
    fig = plt.figure(figsize=(6., 6.))

    grid = ImageGrid(fig, 111,
                    nrows_ncols=(3, 4),
                    axes_pad=0.1)

    for ax, im in zip(grid, rgb_values):
        ax.imshow([[im]])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape((h, w, 4))[:, :, :3]
    return img_array