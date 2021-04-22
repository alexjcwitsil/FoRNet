############################################
## ---- Calculate Feature Statistics ---- ##
###                                      ###
## Given series of values, calculate stats##
## (features).                            ##
############################################

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.stats import norm
import cv2

from skimage import transform

def calc_feature_stats(vals, xy_inds, img_shape):

    ## CURRENT STATS TO BE CALCULATED...
    stat_names = ["sum", "mean", "sd", "skew", "kurt", "area","median", "25_perc", "50_perc", "75_perc"]
    ##stat_names = ["sum", "mean", "sd", "skew", "kurt","median", "25_perc", "50_perc", "75_perc", "max_freq_dist"]


    ## break the x,y points into individual arrays
    x_inds = xy_inds[0]
    y_inds = xy_inds[1]

    ## color features
    val_sum = np.sum(vals)
    val_mean = np.mean(vals)
    val_sd = np.std(vals)
    val_skew = skew(vals)
    val_kurt = kurtosis(vals)
    val_median = np.median(vals)
    val_25perc = np.percentile(vals, 25)
    val_50perc = np.percentile(vals, 50)
    val_75perc = np.percentile(vals, 75)
    
    
    ## spatial features
    val_area = len(x_inds)

    ##############################
    ## GENERATE SEGMENTED IMAGE ##
    ##############################

    seg_img = np.zeros(img_shape[0]*img_shape[1]).reshape(img_shape)
    seg_img[x_inds, y_inds] = vals - val_mean


    ######################
    ## FREQUENCY DOMAIN ##
    ######################

    # ## resize the segmented image for faster "ffting"
    # seg_img_resize = transform.resize(seg_img, (256,256))

    # ##SEG_IMG_UNNORM = np.abs(np.fft.fft2(seg_img))
    # SEG_IMG_UNNORM = np.abs(np.fft.fft2(seg_img_resize))

    # # ## find the sum of the SEGMETNED IMAGE
    # SEG_IMG_sum = np.sum(SEG_IMG_UNNORM)

    # ## normalize the SEGMENTED IMAGE but check if the sum is 0
    # if SEG_IMG_sum == 0:
    #     SEG_IMG = np.fft.fftshift(SEG_IMG_UNNORM)
    # elif SEG_IMG_sum != 0:
    #     SEG_IMG = np.fft.fftshift(SEG_IMG_UNNORM/np.sum(SEG_IMG_UNNORM))
    # #

    # ## find the (first) indicies associated with the max value in the SEG_IMAGE
    # max_freq_inds = list(np.where(SEG_IMG == np.max(SEG_IMG)))

    # ## if no max frequency (in the case of very small segments) set to 0
    # if len(max_freq_inds[0]) == 0:
    #     max_freq_inds[0] = 0
    #     max_freq_inds[1] = 0
    # elif len(max_freq_inds[0]) > 0:
    #     max_freq_inds[0] = max_freq_inds[0][0]
    #     max_freq_inds[1] = max_freq_inds[1][0]
    # #

    # ## find the middle of the FREQUENCY DOMAIN IMAGE
    # img_mid = [round(f/2) for f in img_shape]

    # ## distance between origin and the max indices
    # max_ind_dist = np.sqrt((max_freq_inds[0]-img_mid[0])**2 + (max_freq_inds[1]-img_mid[1])**2)

    # ## apply cumulative sum to rows and columsn
    # ##CUMSUM_SEG_IMG = np.apply_along_axis(np.cumsum,0,SEG_IMG)
    # ##CUMSUM_SEG_IMG = np.apply_along_axis(np.cumsum,1,CUMSUM_SEG_IMG)


    ## save the current stats as a list 
    stats_list = [val_sum, val_mean, val_sd, val_skew, val_kurt, val_area, val_median, val_25perc, val_50perc, val_75perc]
    ##stats_list = [val_sum, val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc, max_ind_dist]


    ## convert the list to a pd DataFrame
    stats_df = pd.DataFrame([stats_list], columns=stat_names)
    
    
    return(stats_df)

