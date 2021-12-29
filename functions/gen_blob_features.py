######################################
## ---- Generate Blob Features ---- ##
###                                ###
##Given an image and Gaussian sigma ##
## perfor blob detection and feature##
## extraction.                      ##
######################################

import fornet as fn
from skimage.color import rgb2hsv
import numpy as np

def gen_blob_features(img, gaus_sig, chan, img_meta):

    #######################
    # Image Preprocessing #
    #######################

    ## create a grayscaled version of the image
    ##gray_img = img[:,:,chan]
 
    ## transfrom the image to HSV space
    # hsv_img = rgb2hsv(img)
    # gray_img = hsv_img[:,:,chan]

    ## transfer the image to lab space
    from skimage.color import rgb2lab ## good
    hsv_img = rgb2lab(img)
    gray_img = hsv_img[:,:,chan]


    ## scale the image between 0 and 1
    ##img = fn.img_range01(img.copy(), max=255)



    ##################
    # Blob Detection #
    ##################

    gray_img_log = fn.log_bd(gray_img, gaus_sig)


    ######################
    # Find Blobs Regions #
    ######################

    blob_img = fn.find_blobs(gray_img_log)

    
    ######################
    # FEATURE EXTRACTION #
    ######################

    blob_info = fn.extract_blob_features(img, blob_img, img_meta)


    return(blob_info)



