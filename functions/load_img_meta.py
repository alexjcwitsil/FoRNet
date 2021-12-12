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

def load_img_meta(path, gray=True, odd_dims=False):

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

    ## initilize lists to hold image means and standard deviations
    img_means = [0]*img.shape[2]
    img_stds = [0]*img.shape[2]

    ## loop over each image channel to calculate meta information
    i=0
    while i < img.shape[2]:
        chan_mean = np.mean(img[:,:,i])
        chan_std = np.std(img[:,:,i])

        ## save the meta information
        img_means[i] = chan_mean
        img_stds[i] = chan_std

        i=i+1
    #

    ## make a list of meta information
    img_meta_info = [img_means, img_stds]

    return(img_meta_info)



