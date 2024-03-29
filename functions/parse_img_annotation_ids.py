##########################################
## ---- Parse Image Annotation Ids ---- ##
###                                    ###
## Given parsed training information,   ##
## find all annotation ids associated   ##
## with the current image.              ##
##########################################

import numpy as np

def parse_img_annotation_ids(img_file, parsed_training_info):

    ## separate the parsed training info into individual objects
    all_images = parsed_training_info[0]
    all_image_ids = parsed_training_info[1]
    all_annotation_image_ids = parsed_training_info[2]

    ## find the index value of the current image in the annotation data
    ##img_index = np.where(all_images == img_file)
    img_index = np.where([i for i in all_images if img_file in i][0] == all_images)

    ## convert the annotation image index to an int so it can be indexed
    ##img_index = int(img_index[0])
    img_index = int(img_index[0][0])

    ## what is the current images ID in the annotation data
    ## cur_img_id = train_info['images'][cur_img_annotation_index]['id']
    img_id = all_image_ids[img_index]

    ## find where the values of the image ids match the current image id
    img_annotation_ids = np.where(all_annotation_image_ids == img_id)

    return(img_annotation_ids[0])



