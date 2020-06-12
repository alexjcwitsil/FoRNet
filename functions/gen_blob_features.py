######################################
## ---- Generate Blob Features ---- ##
###                                ###
##Given an image and Gaussian sigma ##
## perfor blob detection and feature##
## extraction.                      ##
######################################

import fornet as fn

def gen_blob_features(img, gaus_sig):

    #######################
    # Image Preprocessing #
    #######################

    ## create a grayscaled version of the image
    gray_img = fn.grayscale_img(img)


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

    blob_info = fn.extract_blob_features(img, blob_img)


    return(blob_info)



