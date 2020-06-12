import fornet as fn
import numpy as np
import pandas as pd


def gen_segmented_features(img,
                           cur_img_file,
                           label_info,
                           parsed_label_info):
    
    ## check if the raw image is grayscale or color.
    ## and force it to color (i.e. have 3 dimensions)
    if len(img.shape) == 2:
        col_img = np.expand_dims(img,axis=2)
    elif len(img.shape) == 3:
        col_img = img
    #

    ## initilize a background image (pixel values not planes, bridges, etc)
    bkg_img = np.copy(col_img)

    ## setup an array to hold the segmented image locations
    segment_locs = np.zeros(col_img[:,:,0].size).reshape(col_img.shape[0:2])

    ## Find all the annotations for the current image
    cur_img_annotation_ids = fn.parse_img_annotation_ids(cur_img_file, parsed_label_info)

    ##############################
    ### LOOP OVER ALL SEGMENTS ###
    ##############################

    ## initilize a list to hold all the segemntatino xy indicies
    all_seg_xys = list()
  
    ## loop over all the annotation indices
    i=0
    while i<len(cur_img_annotation_ids):

        ## define the current annotation (label) index
        cur_annotation_ind = cur_img_annotation_ids[i] 

        ###############################
        ### GET THE CURRENT OUTLINE ###
        ###############################

        outline_xys = fn.parse_label_outline(label_info, cur_annotation_ind)

        ## get outline label id
        outline_id = fn.parse_label_id(label_info, cur_annotation_ind)

        ##############################
        ### FIND POINTS IN POLYGON ###
        ##############################

        ## the original
        seg_xys = fn.points_in_polygon(outline_xys)

        ## create a flipped version 
        seg_xys_flip = (seg_xys[1], seg_xys[0])

        ## break the segmented x and ys
        seg_xs = seg_xys[0]
        seg_ys = seg_xys[1]
        
        ## update the background image
        bkg_img[seg_ys, seg_xs] = 0

        ##########################
        ### Extract STATISTICS ### 
        ##########################

        ## loop over each color channel
        j=0
        while j < img.shape[2]:

            ## what is the current color channel image
            cur_img_chan = col_img[:,:,j]

            ## get the segmentation values from the current image
            cur_seg_vals = cur_img_chan[seg_ys, seg_xs]

            ## calculate the current channel feature statistics
            cur_chan_stats = fn.calc_feature_stats(cur_seg_vals, seg_xys_flip, cur_img_chan.shape)

            # set up a prefix to name the columns based on the current channel
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


        ## add the categor id to the last column of the stats dataframe
        cur_stats['category_id'] = outline_id

        ## save the current feature stats
        if i == 0:
            ## initilize the image segmentation stats if first iteration
            img_seg_stats = cur_stats
        elif i > 0:
            ## append the current stats to the all blob stats dataframe
            cur_stats_list = cur_stats.iloc[0].tolist()
            img_seg_stats.loc[i] = cur_stats_list

        ## append the current segmentation xy inds to the all list
        all_seg_xys.append(seg_xys)

        print(f' extracting features from labeled segment {str(i)} out of: {str(len(cur_img_annotation_ids))}\r', end="")
     
        i=i+1

    print()

    #########################################
    ### EXTRACT STATISTICS FROM BKG IMAGE ###
    #########################################
    
    ## find the x and y (segmentation)  values of the background image
    seg_xys = np.where(bkg_img != 0)
    seg_xs = seg_xys[1]
    seg_ys = seg_xys[0]

    ## loop over each color channel
    j=0
    while j < bkg_img.shape[2]:
            
        ## also save the current segementation values
        cur_seg_vals =  bkg_img[:,:,j][seg_ys, seg_xs]

        ## calculate the current channel feature statistics
        cur_chan_stats = fn.calc_feature_stats(cur_seg_vals, seg_xys, bkg_img.shape[0:2])

        # set up a prefix to name the columns based on the current channel
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


    ## add the categor id to the last column of the stats dataframe
    cur_stats['category_id'] = 0 ## Always 0 for background

    ## transform to list so we can add it to the full stat dataframe
    cur_stats_list = cur_stats.loc[0].tolist()


    ## before adding background stats, check if there are any seg stats
    ## sometimes there are images without any segmentations...
    if 'img_seg_stats' in dir():
        img_seg_stats.loc[i] = cur_stats_list
    elif 'img_seg_stats' not in dir():
        img_seg_stats = cur_stats
    

    #add the current stats list to the current image segmentation statistics 
    img_seg_stats.loc[i] = cur_stats_list
    
    ## append the current segmentation xy inds to the all list
    all_seg_xys.append(seg_xys)

    img_seg_info = [img.shape[0:2], img_seg_stats, tuple(all_seg_xys)]

    return(img_seg_info)



