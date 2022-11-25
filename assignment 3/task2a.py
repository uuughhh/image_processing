import numpy as np
import skimage
import utils
import pathlib


from matplotlib import pyplot as plt


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    # historgram
    hist,binRanges = np.histogram(im,bins=256)
    
    # normalization
    hist = np.divide(hist, hist.max())
    
    # Calculate possibilities for all combinations of class division
    # possClass1[n] corresponds to possClass2[n+1] for the same division
    # possClass1[-1]
    possClass1 = np.cumsum(hist)
    possClass2 = np.flip(np.cumsum(np.flip(hist)))
    
    # Get the mean of each bin to represent the bin
    binMeans = np.divide((np.add(binRanges[:-1], binRanges[1:])), 2.)
    # in-class means 
    mean1 = np.cumsum(hist * binMeans) / possClass1
    mean2 = np.flip((np.cumsum(np.flip((hist * binMeans))) / np.flip(possClass2)))
    # Calculate all the between-class variances
    betweenClassVars = possClass1[:-1] * possClass2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Find the index with the maximum between-class variances
    maxVarIndex = np.argmax(betweenClassVars)
    
    threshold = binMeans[:-1][maxVarIndex]

    return threshold
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
