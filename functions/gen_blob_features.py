######################################
## ---- Generate Blob Features ---- ##
###                                ###
##Given an image and Gaussian sigma ##
## perfor blob detection and feature##
## extraction.                      ##
######################################

import fornet as fn

def gen_blob_features(img, gaus_sig):

    ##################
    # Blob Detection #
    ##################

    img_log = fn.log_bd(img, gaus_sig)


    ######################
    # Find Blobs Regions #
    ######################

    blob_img = fn.find_blobs(img_log)


    ######################
    # FEATURE EXTRACTION #
    ######################

    blob_info = fn.extract_blob_features(img, blob_img)


    return(blob_info)



