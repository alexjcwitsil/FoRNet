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

def gen_background_features(blob_info, true_img):

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

    ## binarize the true image such that all segments have value of 1
    true_img_bin = true_img.copy()
    true_img_bin[np.where(true_img> 0)] = 1

    ## what are the column names in the bkg stats dataframe (with cat id)
    bkg_stats_col_name = all_blob_stats.columns.tolist() + ['category_id']
    
    ## initilize a data frame to store cur background img segmentation statistics 
    bkg_img_seg_stats = pd.DataFrame(columns = bkg_stats_col_name)
    
    ## initilize a list to hold all the bkg image segmentation xy indicies
    bkg_img_seg_xys = list()


    ## loop over all blobs
    j=0
    while j < len(all_blob_xys):

        # what is the current blob xy indices
        cur_blob_inds = all_blob_xys[j]

        ## build a binary image populated soley by this blob
        blob_img = np.zeros(true_img.shape)
        blob_img[cur_blob_inds] = 1

        ## find out if there is overlap between the blob image and true image
        overlap_img = blob_img + true_img_bin

        ## define boolean object stating if there is overlap
        overlap_bool = (overlap_img==2).any()

        ## if there is overlap move onto the next blob
        if overlap_bool:
            j=j+1
            #print('skipping ' + str(j))
            continue

        ## if there isn't overlap grab the blob statistics
        cur_blob_stats = all_blob_stats.iloc[j]

        ######################################
        ### APPEND STATISTICS TO DATAFRAME ###
        ######################################
        
        ## add the current stats to the bkg image segmentation statistics 
        bkg_img_seg_stats.loc[j] = cur_blob_stats.tolist() + [bkg_cat_id]
        
        ## append the current segmentation xy inds to the all list
        bkg_img_seg_xys.append(cur_blob_inds)
        ##print(str(i) + ' ' + str(len(seg_xs)))

        #print('file: '+str(i)+' blob ' + str(j) + ' out of: ' +  str(len(all_blob_stats)))
        print(f'blob {str(j)} out of {str(len(all_blob_stats))}  \r', end="")
        j=j+1
    print()

    img_bkg_info = [blob_info[0], bkg_img_seg_stats, bkg_img_seg_xys]

    return(img_bkg_info)


