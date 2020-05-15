import fornet as fn
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def extract_blob_features(raw_img, blob_img):

    ## how many blob are there in the blob image
    blob_total = len(np.unique(blob_img))
    
    # what are the statistics you will extract?
    # col_names = ["sum", "mean", "sd", "skew", "kurt", "area"]

    # define a data frame to store all the blob statistics
    # all_blob_stats = pd.DataFrame(columns=col_names)

    ## define a list to hold all the blob x and y indices
    blob_xy_inds = list()

    # loop over each label
    i = 0
    while(i <= blob_total):

        # what is the current blob label as a boolean matrix
        cur_blob = blob_img == i
        
        # Find the label x and y indices
        # tuple of length 2, which holds x and y label locations
        cur_blob_xys = np.where(cur_blob)

        ## save the current blob xy indices to the list
        blob_xy_inds.append(cur_blob_xys)

        # isolate the current blob color values
        cur_blob_vals = raw_img[cur_blob_xys]

        cur_stats = fn.calc_feature_stats(cur_blob_vals, cur_blob_xys)
        
        # COLOR STATISTICS
        # cur_sum = np.sum(cur_blob_vals)
        # cur_mean = np.mean(cur_blob_vals)
        # cur_sd = np.std(cur_blob_vals)
        # cur_skew = skew(cur_blob_vals)
        # cur_kurt = kurtosis(cur_blob_vals)

        # # spatial STATISTICS
        # cur_area = len(cur_blob_vals)

        # # save the current stats as a list
        # cur_stats = list([cur_sum, cur_mean, cur_sd, cur_skew, cur_kurt, cur_area])
        # # add the current stats list to the blob stats dataframe
        ##all_blob_stats.loc[i] = cur_stats


        ## save the current feature stats
        
        if i == 0:
            ## initilize the image segmentation stats if first iteration
            all_blob_stats = cur_stats
        elif i > 0:
            ## append the current stats to the all blob stats dataframe
            cur_stats_list = cur_stats.iloc[0].tolist()
            all_blob_stats.loc[i] = cur_stats_list

        
        print(f' blob {str(i)} out of {str(blob_total)}\r', end="")
        i = i+1
    print()

    # combine the feature information and the x and y label (blob) locations
    # list of [image shape, blob statistics, blob x,y indices]
    # note the blob xy indices are saved as tuple
    img_blob_info = [raw_img.shape, all_blob_stats, tuple(blob_xy_inds)]

    return(img_blob_info)



