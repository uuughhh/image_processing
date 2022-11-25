import utils
import numpy as np


def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    def neighboringPoints (row:int,col:int)-> list:
        """
        Returning 8 neighboring points of the seed points.
        """
        neighbors = [
            [row-1,col-1],
            [row-1,col],
            [row-1,col+1],
            [row,col-1],
            [row,col+1],
            [row+1,col-1],
            [row+1,col],
            [row+1,col+1]
            ]
        return neighbors

    def inspect (row,col,rowN,colN) -> bool:
        """
        decide whethter a point can be marked as true
        mark true to segmented image if valid
        """
        if abs(im[row,col] - im[rowN,colN]) <= intensity_threshold and segmented[rowN,colN]!=True:
            # set true if the absolute difference is smaller than threshold and never visited
            segmented[rowN,colN] = True
            return True
        # not a valid point
        return False

    def recursiveGrowing (candidates:list,row:int,col:int) :
        """
        A recursive function to grow until the candidates list is empty
        """
        if len(candidates) == 0:
            # no more points to inspect
            return
        else:
            for rowN,colN in candidates:
                # pixel within range
                if 0 <= rowN < im.shape[0] and 0 <= rowN < im.shape[1]:
                    # a new point satisfying the predicate found
                    if inspect(row,col,rowN,colN) == True:
                        new_candidates = neighboringPoints(rowN,colN)
                        recursiveGrowing(new_candidates,row,col)
            return


    segmented = np.zeros_like(im).astype(bool)
    im = im.astype(float)
    for row, col in seed_points:
        segmented[row, col] = True
        # iterate through neighbors
        candidates = neighboringPoints(row,col)
        # start recursive growing
        recursiveGrowing(candidates,row,col)

    return segmented
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [  # (row, column)
        [254, 138],  # Seed point 1
        [253, 296],  # Seed point 2
        [233, 436],  # Seed point 3
        [232, 417],  # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
        im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
        segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)
