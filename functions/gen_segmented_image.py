#########################################
## ---- Generate Segmented Image  ---- ##
###                                   ###
#########################################

import fornet as fn
import numpy as np
import pandas as pd

def gen_segmented_image(seg_info):
    
    ## isolate the labeled data features
    seg_stats = seg_info[1]

    ## what is the image shape
    img_shape = seg_info[0]

    ## grab the categor ids column
    cat_ids = seg_stats.iloc[:,seg_stats.shape[1]-1]



    ### THIS COULD BE A PROBLEM OR A HUGE SOLUTION ###
    ##cat_ids = cat_ids + 1 
    ### THIS COULD BE A PROBLEM OR A HUGE SOLUTION ###







    ## gray all the segmented x and y indicies (locations)
    seg_xys = seg_info[2]

    ## initilize the segmented image
    seg_img = np.zeros(img_shape[0] * img_shape[1]).reshape(img_shape)

    # loop over all the category ids to build a segmented image
    # recall we don't need to plot the background... for now...
    i=0
    while i<(len(cat_ids)-1):

        # current category
        cur_cat = cat_ids.iloc[i]

        ## current segment xs
        cur_xys = seg_xys[i]

        ## populate the segmented image with the current category id at the current xy locations
        seg_img[cur_xys[1],cur_xys[0]] = cur_cat

        i=i+1


    return(seg_img)
















