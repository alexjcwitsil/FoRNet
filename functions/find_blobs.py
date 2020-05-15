import cv2
import numpy as np

def find_blobs(bd_img):

    ######################################
    ## BINARIZE THE BLOB DETECTED IMAGE ##
    ######################################

    # Binarize both a 'positive' and 'negative' version of the log img
    ret, img_bin_pos = cv2.threshold(bd_img, 0, 1, cv2.THRESH_BINARY)
    ret, img_bin_neg = cv2.threshold(bd_img*-1, 0, 1, cv2.THRESH_BINARY)

    # convert the positive and negative binary images back to int
    img_bin_pos = np.array(img_bin_pos, dtype=np.uint8)
    img_bin_neg = np.array(img_bin_neg, dtype=np.uint8)


    ########################
    # CONNECTED COMPONENTS #
    ########################

    # specify either 4 or 8 connectivity
    connectivity = 4
    img_cca_pos = cv2.connectedComponentsWithStats(img_bin_pos, connectivity)
    img_cca_neg = cv2.connectedComponentsWithStats(img_bin_neg, connectivity)

    # break the output tuples into indidual blob location cells
    blob_locs_pos = img_cca_pos[1]
    blob_locs_neg = img_cca_neg[1]

    # normalize the negative array labels relative to the raw labels
    blob_locs_norm = (blob_locs_neg + np.max(blob_locs_pos)) * img_bin_neg

    # combine the raw and negative connected components
    blob_locs = blob_locs_pos + blob_locs_norm

    return(blob_locs)
