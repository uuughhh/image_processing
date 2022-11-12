import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)

    # get |F{f}|
    def amplitude(fft):
        real = fft.real
        imag = fft.imag
        return np.sqrt(real**2 + imag**2)

    # get frequency doamin of the iamge
    fft = np.fft.fft2(im)
    fft_v = np.fft.fftshift(fft)
    fft_v = np.log(amplitude(fft_v)+1)  # visualize frequency domain

    height = fft.shape[0]
    width = fft.shape[1]
    kernel_size = kernel.shape[0]

    # pad kernel to the same size as the image
    pad_kernel = np.pad(kernel,[(height//2, height//2-kernel_size), (width//2, width//2-kernel_size)])
    # transform to frequency domain
    fft_kernel = np.fft.fft2(pad_kernel)
    fft_kernel_v = np.fft.fftshift(fft_kernel)
    fft_kernel_v = np.log(amplitude(fft_kernel_v)+1)    # visualize kernel in frequency domain

    # apply filter to the image
    fft_filtered = fft * fft_kernel
    fft_filtered_v = np.fft.fftshift(fft_filtered)
    fft_filtered_v = np.log(amplitude(fft_filtered_v)+1)    #visualize filtered image in frequency domain

    conv_result = np.fft.ifftshift(np.fft.ifft2(fft_filtered)).real #visualize result

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        # Visualize FFT
        plt.imshow(fft_v)
        plt.subplot(1, 5, 3)
        # Visualize FFT kernel
        plt.imshow(fft_kernel_v)
        plt.subplot(1, 5, 4)
        # Visualize filtered FFT image
        plt.imshow(fft_filtered_v)
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
