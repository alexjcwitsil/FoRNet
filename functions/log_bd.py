import fornet as fn
import cv2
import numpy as np

def log_bd(img, sig):

    ## filter with a Gaussian kernal
    ## this is WAY too slow
    ##img_gaus_filt = cv2.GaussianBlur(img, img.shape[0:2], sig)

    ## build a gaussian
    gaus_mask = fn.build_gaus2d(img.shape[0],img.shape[1],sig)

    ## filter with the Gaussian kernal
    img_gaus_filt = cv2.filter2D(img, -1, gaus_mask)

    ## Laplacian of Gaussain
    img_log = cv2.Laplacian(img_gaus_filt, -1)

    ## return the LoG image
    return(img_log)

