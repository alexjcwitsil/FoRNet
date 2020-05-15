import fornet as fn
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def extract_segmented_features(raw_img, segmented_img):

    ## how many segments are there in the labeled image
    seg_total = len(np.unique(segmented_img))

    ## define a list to hold all the segmented x and y indices
    seg_xy_inds = list()

    # loop over each label
    i = 0
    while(i <= seg_total):

        # what is the current segment label as a boolean matrix
        cur_seg = segmented_img == i
        
        # Find the label x and y indices
        # tuple of length 2, which holds x and y label locations
        cur_seg_xys = np.where(cur_seg)

        ## save the current seg xy indices to the list
        seg_xy_inds.append(cur_seg_xys)

        # isolate the current segmented color values
        cur_seg_vals = raw_img[cur_seg_xys]

        cur_stats = fn.calc_feature_stats(cur_seg_vals, cur_seg_xys)
        
        ## save the current feature stats
        
        if i == 0:
            ## initilize the image segmentation stats if first iteration
            all_seg_stats = cur_stats
        elif i > 0:
            ## append the current stats to the all segmented stats dataframe
            cur_stats_list = cur_stats.iloc[0].tolist()
            all_seg_stats.loc[i] = cur_stats_list

        
        print(f' segment {str(i)} out of {str(seg_total)}\r', end="")
        i = i+1
    print()

    # combine the feature information and the x and y label (segment) locations
    # list of [image shape, segment statistics, segment x,y indices]
    # note the segment xy indices are saved as tuple
    img_seg_info = [raw_img.shape, all_seg_stats, tuple(seg_xy_inds)]

    return(img_seg_info)


