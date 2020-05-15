import fornet as fn
import numpy as np


def find_segments(img, cur_img_file, train_info, parsed_training_info):

    ## setup an array to hold the segmented image locations
    segment_locs = np.zeros(img.size).reshape(img.shape)

    ## Find all the annotations for the current image
    cur_img_annotation_ids = fn.parse_img_annotation_ids(cur_img_file, parsed_training_info)

    ##############################
    ### LOOP OVER ALL SEGMENTS ###
    ##############################
  
    ## loop over all the annotation indices
    i=0
    while i<len(cur_img_annotation_ids):

        ## define the current annotation (label) index
        cur_annotation_ind = cur_img_annotation_ids[i] 

        ###############################
        ### GET THE CURRENT OUTLINE ###
        ###############################

        outline_xys = fn.parse_label_outline(train_info, cur_annotation_ind)

        ## get outline label id
        outline_id = fn.parse_label_id(train_info, cur_annotation_ind)

        ##############################
        ### FIND POINTS IN POLYGON ###
        ##############################

        seg_xys = fn.points_in_polygon(outline_xys)

        ## break the segmented x and ys
        seg_xs = seg_xys[0]
        seg_ys = seg_xys[1]

        ## populate the segmented image xy locations with the outline id
        segment_locs[seg_ys, seg_xs] = outline_id

        i=i+1
    
    return(segment_locs)


