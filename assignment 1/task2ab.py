import matplotlib.pyplot as plt
import pathlib

import numpy
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    
    im = numpy.dot(im[...,:3], [0.212,0.7152, 0.0722])

    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    im = 1 - im


    return im

im_inverse = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_inverse.jpg"), im_inverse, cmap="gray")
plt.imshow(im_inverse, cmap="gray")