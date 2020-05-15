##################################################
## ---- Join Label and Background Features ---- ##
###                                            ###
## Read in backgound and label features for each##
## image and join them into a singel feature    ##
## list structure.                              ##
##################################################

#Dependencies
import numpy as np
import pandas as pd

def join_seg_bkg_features(seg_info, bkg_info):

    ## separate the segmentation and background feature information
    seg_features = seg_info[1]
    bkg_features = bkg_info[1]

    ## convert these pandas dataframes to numpy arrays
    seg_features_array = seg_features.to_numpy()
    bkg_features_array = bkg_features.to_numpy()

    ## join/stack these arrays 
    img_features_array = np.vstack((seg_features_array, bkg_features_array))

    ## turn this array back into a pandas dataframe
    img_features = pd.DataFrame(img_features_array)

    ## add the column names back on
    img_features.columns = seg_features.columns

    return(img_features)


