###################################
## ---- Parse Training Info ---- ##
###                             ###
## Given a training information  ##
## object parse (extract) some   ##
## useful information.           ##
###################################

import numpy as np

def parse_training_info(training_info):

    ## make a list of all the annotation image files and ids
    all_images = []
    all_image_ids = [] 
    i=0
    while i < len(training_info['images']):
        all_images.append(training_info['images'][i]['file_name'])
        all_image_ids.append(training_info['images'][i]['id'])
        i=i+1

    ## convert all annotation image file names to np array
    all_images = np.array(all_images)

    ## separate the annotations out
    ## this is a LIST with each list element being a dictionary
    all_annotations = training_info['annotations']

    ## set up a list to hold the image ids
    all_annotation_image_ids = np.zeros(len(all_annotations))

    i=0
    while i < len(all_annotations):
        #grab and save the image ID for this iteration 
        all_annotation_image_ids[i] = int(all_annotations[i]['image_id'])

        i=i+1

    ## combine all information into a list
    parsed_info = [all_images, all_image_ids, all_annotation_image_ids]

    return(parsed_info)
