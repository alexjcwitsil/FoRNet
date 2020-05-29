###############
### TESTING ###
###############

import fornet as fn
import os
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.models import load_model



##############
### INPUTS ###
##############
## how many guassian sigmas to use
## note these must be in a list
gaus_sigs = [8, 10, 15, 20 ] 

## where is the project directory
## note there must be data correctly stored within the project sub-directories
project_path = '/home/alexwitsil/projects/isaid_imagery'


#################################
### READ IN IMAGE ANNOTATIONS ###
#################################

## read in the training information 
with open(project_path + '/data/iSAID_train.json') as f:
    label_info = json.load(f)
    ## keys -> 'images', 'categories', 'annotations'


###########################
## PARSE ANNOTATION INFO ##
###########################

parsed_label_info = fn.parse_label_info(label_info)


##########################
### FIND THE TEST DATA ###
##########################

## what is the testing image path
test_img_path = project_path + '/data/testing/raw_images/'

## list the image files
img_files = os.listdir(test_img_path)


############################
### LOOP OVER THE IMAGES ###
############################

i=0
while i < len(img_files):

    ## grab the current file
    cur_img_file = img_files[i]
    img_path = test_img_path + cur_img_file

    ## load in the current image
    img = fn.load_image(img_path, odd_dims=True)

    
    #################################
    ## GENERATE SEGMENTED FEATURES ##
    #################################

    img_seg_info = fn.gen_segmented_features(img, cur_img_file, label_info, parsed_label_info)


    ##############################
    ## GENERATE SEGMENTED IMAGE ##
    ##############################

    true_img_seg = fn.gen_segmented_image(img_seg_info)


    #####################################
    ### LOOP OVER THE GAUSSIAN SIGMAS ###
    #####################################

    ## initilize a list to hold labeled and probability images
    labeled_img_list = []
    probability_img_list = []

    j=0
    while j < len(gaus_sigs):

        ## what is the current Gaussian sigmas
        cur_sig = gaus_sigs[j]

        
        #################
        ## READ IN ANN ##
        #################

        ## Where is the model and scaling factors 
        model_dir = project_path + '/results/nn_models/sig' + str(cur_sig) + '/'
        model_file = 'vanilla_model.h5'
        scaling_factors_file = 'scaling_factors.pickle'

        ## load the model 
        model = load_model(model_dir + model_file)

        ## load the scaline factors
        with open(model_dir + scaling_factors_file,'r+b') as handle:
            sc = pickle.load(handle)

        
        ############################
        ## GENERATE BLOB FEATURES ##
        ############################

        img_blob_info = fn.gen_blob_features(img, cur_sig)

        ## separate the blob features
        blob_features_unnorm = img_blob_info[1]

        ## remove any nan values
        blob_features_unnorm = blob_features_unnorm[~np.isnan(blob_features_unnorm).any(axis=1)]

        # scale the features
        cur_features = sc.fit_transform(blob_features_unnorm)


        ########################
        ### MAKE PREDICTIONS ###
        ########################

        ## make predictions using the model
        labels_pred = model.predict(cur_features)

        #Converting predictions to label (opposite of OHE)
        pred = list()

        # also create a list of prediction probabilities
        pred_probs = list()

        for k in range(len(labels_pred)):
            pred.append(np.argmax(labels_pred[k]))
            pred_probs.append(np.max(labels_pred[k]))

        # build a labeled image and probability image
        labeled_img = np.zeros(img_blob_info[0])
        probability_img = np.zeros(img_blob_info[0])


        # loop over the blob indices
        k=1
        while k < len(pred):

            # isolate the current blob indices
            blob_xs = img_blob_info[2][k][0]
            blob_ys = img_blob_info[2][k][1]

            # populate these indices with the current label
            labeled_img[blob_xs,blob_ys] = pred[k]
            probability_img[blob_xs,blob_ys] = pred_probs[k]

            k=k+1

        ## add current labeled image to the list
        labeled_img_list.append(labeled_img)
        probability_img_list.append(probability_img)


        ## DME ##
        plt.subplot(1,2,1)
        plt.imshow(true_img_seg)
        plt.subplot(1,2,2)
        plt.imshow(labeled_img)
        

        
        
        j=j+1


    ###############################
    ### STACK AND REDUCE LABELS ###
    ###############################

    pred_img_seg = fn.stack_reduce_labels(labeled_img_list, probability_img_list)
    
    ## DME ##
    plt.subplot(1,2,1)
    plt.imshow(true_img_seg)
    plt.subplot(1,2,2)
    plt.imshow(pred_img_seg)

    
    
    ################
    ### EVALUATE ###
    ################

    
    






