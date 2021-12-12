##########################
## ---- LOAD IMAGE ---- ##
###                    ###
## Load image with certa-#
#-in color channels and ##
## decide if to have odd##
## dimensions, and float##
##########################

import cv2
import numpy as np
import fornet as fn

def load_image(path, gray=True, odd_dims=False, norm_chans=False):

    ## do you want a grayscaled image or RGB color
    if gray==True:
        img_type = cv2.IMREAD_GRAYSCALE
    elif gray==False:
        img_type = cv2.IMREAD_COLOR

    # read in raw image
    img_raw = cv2.imread(path, img_type)

    ## do you want to force odd dimensions?
    if odd_dims==True:

        # check to see if the image is odd or even... force it to be odd
        if(img_raw.shape[0] % 2 == 0):
            img_raw = np.delete(img_raw, -1, 0)

        if(img_raw.shape[1] % 2 == 0):
            img_raw = np.delete(img_raw, -1, 1)

    # convert the image from int (I think) to a float
    img = np.float64(img_raw)

    
    ## do you want to normalize each channel by subtracting the mean and dividing by STD
    if norm_chans==True:
        img = fn.norm_chans(img)
    #


    return(img)

