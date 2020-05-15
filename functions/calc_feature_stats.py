############################################
## ---- Calculate Feature Statistics ---- ##
###                                      ###
## Given series of values, calculate stats##
## (features).                            ##
############################################

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def calc_feature_stats(vals, xy_inds):

    ## CURRENT STATS TO BE CALCULATED...
    stat_names = ["sum", "mean", "sd", "skew", "kurt", "area"]

    ## break the x,y points into individual arrays
    x_inds = xy_inds[0]
    y_inds = xy_inds[1]

    ## color features
    val_sum = np.sum(vals)
    val_mean = np.mean(vals)
    val_sd = np.std(vals)
    val_skew = skew(vals)
    val_kurt = kurtosis(vals)
    
    ## spatial features
    val_area = len(x_inds)

    ###########################
    ## GENERATE BINARY IMAGE ##
    ###########################

    # ## generate the segmented image as a binary and original valued img
    # cur_seg_img = np.zeros(img.size).reshape(img.shape)
    # cur_seg_bin = np.zeros(img.size).reshape(img.shape)

    # ## populate the segemented image
    # ## NOTE WE DONT USE THESE YET BUT THE YS AND XS ARE SWITCHED
    # cur_seg_img[seg_ys, seg_xs] = img[seg_ys, seg_xs]
    # cur_seg_bin[seg_ys, seg_xs] = 1

    ## save the current stats as a list 
    stats_list = [val_sum, val_mean, val_sd, val_skew, val_kurt, val_area]

    ## convert the list to a pd DataFrame
    stats_df = pd.DataFrame([stats_list], columns=stat_names)
    
    
    return(stats_df)

