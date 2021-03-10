import fornet as fn
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def extract_blob_features(raw_img, blob_img):

    ## check if the raw image is grayscale or color.
    ## and force it to color (i.e. have 3 dimensions)
    if len(raw_img.shape) == 2:
        col_img = np.expand_dims(raw_img,axis=2)
    elif len(raw_img.shape) == 3:
        col_img = raw_img
    #

    ## how many blob are there in the blob image
    blob_total = len(np.unique(blob_img))

    ## define a list to hold all the blob x and y indices
    blob_xy_inds = list()

    # loop over each label
    i = 1 # NOTE WE START AT 1 HERE! 
    while i <= blob_total:

        # what is the current blob label as a boolean matrix
        cur_blob = blob_img == i
        
        # Find the label x and y indices
        # tuple of length 2, which holds x and y label locations
        cur_blob_xys = np.where(cur_blob)

        ## if the blob is empty, skip it!
        if len(cur_blob_xys[0]) == 0:
            i=i+1
            continue
        #
            
        ## save the current blob xy indices to the list
        blob_xy_inds.append(cur_blob_xys)

        ## now loop over each color channel
        j=0
        while j < col_img.shape[2]:

            ## what is the current color channel image
            cur_img_chan = col_img[:,:,j]

            # isolate the current blob color values
            cur_blob_vals = cur_img_chan[cur_blob_xys]

            ## calculate the feature statistics
            cur_chan_stats = fn.calc_feature_stats(cur_blob_vals, cur_blob_xys,cur_img_chan.shape)
            ## set up a prefix to name the columns based on the current channel
            chan_name = 'chan' + str(j) + '_'

            ## add the channel prefix to the dataframe's column names
            col_names = [chan_name + f for f in cur_chan_stats.columns]

            ## rename the current channel columns
            cur_chan_stats.columns = col_names

            ## add this channel's stats to the image stats
            if j==0:
                cur_stats = cur_chan_stats
            elif j>0:
                cur_stats = pd.concat([cur_stats,cur_chan_stats],axis=1)
            #

            j=j+1
        #
        


        ## save the current feature stats
        
        if i == 1:
            ## initilize the image segmentation stats if first iteration
            all_blob_stats = cur_stats
        elif i > 1:
            ## append the current stats to the all blob stats dataframe
            cur_stats_list = cur_stats.iloc[0].tolist()
            all_blob_stats.loc[i] = cur_stats_list

        
        ##print(f' Extracting features from blob {str(i)} out of {str(blob_total)}\r', end="")
        i = i+1
    print()

    # combine the feature information and the x and y label (blob) locations
    # list of [image shape, blob statistics, blob x,y indices]
    # note the blob xy indices are saved as tuple
    img_blob_info = [col_img.shape[0:2], all_blob_stats, tuple(blob_xy_inds)]

    return(img_blob_info)



