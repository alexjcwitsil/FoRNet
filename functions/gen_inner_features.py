############################################
## ---- Generate Background Features ---- ##
###                                      ###
## Read in blob info and true image segmen##
##-tations  and find all blobs that do not##
## overlap. Return these statistics as    ##
## background features.                   ##
############################################


import numpy as np
import pandas as pd

def gen_inner_features(blob_info, true_img):

    ## what is the category id of the background 
    bkg_cat_id = 0 ## this should be same for all scripts

    ## separate the features from blob indices
    all_blob_stats = blob_info[1]
    all_blob_xys = blob_info[2]

    ## FORCE TRUE SEGMENTATION TO HAVE ODD DIMS
    ## recall predicted images are forced to have odd dimensions
    ## therefore we need to force same constraint on true images
    if(true_img.shape[0] % 2 == 0): 
        true_img = np.delete(true_img,-1,0)
    if(true_img.shape[1] % 2 == 0):
        true_img = np.delete(true_img,-1,1)

    ## what are the column names in the bkg stats dataframe (with cat id)
    inner_stats_col_name = all_blob_stats.columns.tolist() + ['category_id']

    ## initilize a data frame to store cur label img segmentation statistics 
    inner_img_seg_stats = pd.DataFrame(columns = inner_stats_col_name)

    ## initilize a list to hold all the inner image segmentation xy indicies
    inner_img_seg_xys = list()

    ## how many unique segment id labels are there in the image
    unique_labs = np.unique(true_img)

    ## loop over each unique label
    i=0
    while i < len(unique_labs):

        ## what is the current unique label
        cur_lab = unique_labs[i]
        
        ## binarize the true image such that only the current label has value of 1
        true_img_bin = np.zeros(true_img.size).reshape(true_img.shape)
        true_img_bin[np.where(true_img == cur_lab)] = 1

        ## create an object to keep track of the number of inner features
        n_inner_features = 0

        ## loop over all blobs
        j=0
        while j < len(all_blob_xys):

            # what is the current blob xy indices
            cur_blob_inds = all_blob_xys[j]

            ## build a binary image populated soley by this blob
            blob_img = np.zeros(true_img.shape)
            blob_img[cur_blob_inds] = 1

            ## find out if there is overlap between the blob image and true image
            overlap_img = (blob_img * true_img_bin) + blob_img

            ## check if there is any overlap at all btwn the blob and current segment
            any_overlap = (overlap_img > 1).any()

            ## check if the the blob is completely within the current segment
            blob_fully_in_segment = (overlap_img == 1).any() == False

            ## put these two check together
            inner_blob_check = any_overlap & blob_fully_in_segment

            ## define boolean object stating if there is overlap
            ##overlap_bool = (overlap_img < 2).any()

            ## if the blob is not completely within the segmented area
            ## or there is no overlap whatsoever, move onto the next blob
            if inner_blob_check == False:
                j=j+1
                #print('skipping ' + str(j-1))
                continue

            #print('moving onto ' + str(j))
            #j=j+1
            n_inner_features = n_inner_features + 1 

            ## if the blob is completetly within the current segment, grab the blob statistics
            cur_blob_stats = all_blob_stats.iloc[j]

            
            ######################################
            ### APPEND STATISTICS TO DATAFRAME ###
            ######################################

            ## add the current stats to the inner image segmentation statistics 
            inner_img_seg_stats.loc[j] = cur_blob_stats.tolist() + [int(cur_lab)]

            ## append the current segmentation xy inds to the all list
            inner_img_seg_xys.append(cur_blob_inds)
            ##print(str(i) + ' ' + str(len(seg_xs)))

            #print('file: '+str(i)+' blob ' + str(j) + ' out of: ' +  str(len(all_blob_stats)))
     
            j=j+1

        print(f' generated {str(n_inner_features)} inner features from label {str(cur_lab)} out of {str(len(unique_labs))}  \r', end="")
        i=i+1
    #
    print()
        

    img_inner_info = [blob_info[0], inner_img_seg_stats, inner_img_seg_xys]

    return(img_inner_info)


