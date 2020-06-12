##################################################
## ---- Cross Correlation in 2 Dimensions ----- ##
##################################################

import numpy as np


def xcorr2d(img1, img2, norm = True):

    ## remove the mean
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)    

    ## go to frequency domain
    IMG1 = np.fft.fft2(img1)
    IMG2c = np.conj(np.fft.fft2(img2))

    ## take cross correlation in frequency domain
    R = IMG1 * IMG2c

    ## go back into spatial domain and adjust the quandrants
    r = np.fft.fftshift(np.real(np.fft.ifft2(R)/len(R)))

    ## normalize the correlation values
    if norm: 
        r = r/len(img1)/(np.std(img1) * np.std(img2))

    return(r)


