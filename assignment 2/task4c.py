import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)
    def amplitude(fft):
        real = fft.real
        imag = fft.imag
        return np.sqrt(real**2 + imag**2)


    fft = np.fft.fft2(im)

    h = fft.shape[0]
    w = fft.shape[1]
    for i in range(h):
        for j in range(w):
            if fft[i,j] >3:
                fft[i,j] = 3
            elif fft[i,j]<-3:
                fft[i,j] = -3

    fft_v = np.fft.fftshift(fft)
    fft_v = np.log(amplitude(fft_v)+1)
    result = np.fft.ifft2(fft).real

    
    # plt.subplot(1, 2, 1)
    # plt.imshow(fft_v)
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()

    F2 = np.fft.fft(im)[207]
    plt.plot(abs(F2))
    plt.ylim(0,1)
    plt.show()

    im_filtered = result

    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
