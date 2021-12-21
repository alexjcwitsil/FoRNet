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
import fornet as fn

from skimage import transform

def calc_feature_stats(vals, xy_inds, img_shape, img_mean, img_std):

    ## CURRENT STATS TO BE CALCULATED...
    ##stat_names = ["mean", "sd", "skew", "kurt", "median", "25_perc", "50_perc", "75_perc"]
    ##stat_names = ["sum", "mean", "sd", "skew", "kurt", "area","median", "25_perc", "50_perc", "75_perc"]
    ## stat_names = ["sum", "mean", "sd", "skew", "kurt","median", "25_perc", "50_perc", "75_perc", "max_freq_dist"]

    ##stat_names = ["mean", "sd", "skew", "kurt", "median", "25_perc", "50_perc", "75_perc", "texture_contrast", "texture_dissimilarity", "texture_homogeneity", "texture_asm", "texture_energy", "texture_correlation"]
    stat_names = ["sum", "mean", "sd", "skew", "kurt", "median", "25_perc", "50_perc", "75_perc", "texture_contrast", "texture_dissimilarity", "texture_homogeneity", "texture_correlation"]
    ##stat_names = ["mean", "sd", "skew", "kurt", "median", "25_perc", "50_perc", "75_perc"]
    ##stat_names = ["mean", "sd", "skew", "kurt", "median", "25_perc", "50_perc", "75_perc", "texture_dissimilarity"]


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
    ##seg_img[x_inds, y_inds] = vals
    ## seg_img[x_inds, y_inds] = fn.range01(vals)*255
    ## seg_vals = vals - np.min(vals)
    seg_vals = (vals * img_std) + img_mean #- np.min(vals)
    seg_img[x_inds, y_inds] = seg_vals
    seg_img = seg_img.astype(int)


    ###############
    ### TEXTURE ###
    ###############
    
    ## crop the segmented image
    ## crop_seg = seg_img[np.min(x_inds):np.max(x_inds)+1, np.min(y_inds):np.max(y_inds)+1]

    


    ## try normalizing the cropped segmented image between 0 and 255
    ##crop_seg = fn.range01(crop_seg)*255
    ## fornce cropped segmented image to be postive (shift it to positive values). 
    ##crop_seg = crop_seg - np.min(crop_seg)
 

    from skimage.feature import greycomatrix, greycoprops

    ## generate the GCLM matrix
    ##gclm_all = greycomatrix(crop_seg.astype(int), [1], [0], levels=256,symmetric=True, normed=False)
    gclm_all = greycomatrix(seg_img, [1], [0], levels=256,symmetric=True, normed=False)
    

    ## remove the zero row and column
    gclm_dme = np.delete(gclm_all, 0, axis=0)
    gclm_unnorm = np.delete(gclm_dme, 0, axis=1)

    ## textures don't depend on nomralization.  
    gclm = gclm_unnorm

    ## normalize
    ##gclm = (gclm_unnorm - np.min(gclm_unnorm)) / (np.max(gclm_unnorm) - np.min(gclm_unnorm))
    ##gclm = gclm_unnorm / np.sum(gclm_unnorm)

    ## calculate texture features
    texture_contrast = greycoprops(gclm, 'contrast')[0][0]
    texture_dissimilarity = greycoprops(gclm, 'dissimilarity')[0][0]
    texture_homogeneity = greycoprops(gclm, 'homogeneity')[0][0]
    texture_asm = greycoprops(gclm, 'ASM')[0][0]
    texture_energy = greycoprops(gclm, 'energy')[0][0]
    texture_correlation = greycoprops(gclm, 'correlation')[0][0]

    # import random
    # rand_val = random.randint(0,255)
    # texture_contrast = random.randint(0,255)
    # texture_dissimilarity = random.randint(0,255)
    # texture_homogeneity = random.randint(0,255)
    # texture_asm = random.randint(0,255)
    # texture_energy = random.randint(0,255)
    # texture_correlation = random.randint(0,255)


    # ######################
    # ## FREQUENCY DOMAIN ##
    # ######################

    ## normalize the segmented image
    # seg_img = np.zeros(img_shape[0]*img_shape[1]).reshape(img_shape)
    # seg_img[x_inds, y_inds] = vals - val_mean


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

    # ## frequency domain statistical moments
    # VAL_SD = np.std(SEG_IMG)
    # VAL_SKEW = skew(skew(SEG_IMG))
    # VAL_KURT = kurtosis(kurtosis(SEG_IMG))
    # VAL_MEDIAN = np.median(SEG_IMG)
    # VAL_25PERC = np.percentile(SEG_IMG, 25)
    # VAL_50PERC = np.percentile(SEG_IMG, 50)
    # VAL_75PERC = np.percentile(SEG_IMG, 75)

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
    ##stats_list = [val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc]
    ##stats_list = [val_sum, val_mean, val_sd, val_skew, val_kurt, val_area, val_median, val_25perc, val_50perc, val_75perc]


    
    ##stats_list = [val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc, texture_contrast, texture_dissimilarity, texture_homogeneity, texture_asm, texture_energy, texture_correlation]
    stats_list = [val_sum, val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc, texture_contrast, texture_dissimilarity, texture_homogeneity, texture_correlation]
    ##stats_list = [val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc]
    ##stats_list = [val_mean, val_sd, val_skew, val_kurt, val_median, val_25perc, val_50perc, val_75perc, texture_dissimilarity]



    ## convert the list to a pd DataFrame
    stats_df = pd.DataFrame([stats_list], columns=stat_names)
    
    
    return(stats_df)

