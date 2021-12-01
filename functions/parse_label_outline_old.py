###################################
## ---- Parse Label Outline ---- ##
###                             ###
## Grab the x and y outline values#
## from the training information ##
###################################

import numpy as np

def parse_label_outline(train_info, index):

    ## get the current annotation
    cur_annotation = train_info["annotations"][index]

    ## keys -> 'id', 'image_id', 'segmentation', 'category_id',
    ## 'category_name', 'iscrowd', 'area', 'bbox'

    ## go ahead and save the current annotations category id and name
    cur_annotation_category = cur_annotation['category_name']
    cur_annotation_category_id = cur_annotation['category_id']

    ## isolate the OUTLINE for the current segmentation
    ## ndarray with structure ([x1, y1, x2, y2, ..., xn, yn])
    cur_outline = np.array(cur_annotation["segmentation"][0])

    ## find the x and y indices in the segmentation array
    outline_inds = np.arange(len(cur_outline))+1
    outline_x_inds = np.array(outline_inds % 2, dtype=bool)
    outline_y_inds = np.array((outline_inds + 1) % 2, dtype=bool)

    ## separate the segmentation xs and ys
    outline_xs = cur_outline[outline_x_inds]
    outline_ys = cur_outline[outline_y_inds]

    ## set up an array with outline xy values in columns
    outline_xys = np.column_stack((outline_xs,outline_ys))

    return(outline_xys)

