###############################
## ---- Parse Label ID  ---- ##
###                         ###
## Grab the x and y outline id#
###############################

import numpy as np

def parse_label_id(train_info, index):

    ## get the current annotation
    cur_annotation = train_info["annotations"][index]

    ## grab the category id key
    cur_annotation_id = cur_annotation['category_id']

    return(cur_annotation_id)

