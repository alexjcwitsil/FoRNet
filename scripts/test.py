###############
### TESTING ###
###############

import fornet as fn
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model



##############
### INPUTS ###
##############
## how many guassian sigmas to use
## note these must be in a list
gaus_sigs = [8, 10] 

## where is the project directory
## note there must be data correctly stored within the project sub-directories
project_path = '/home/alexwitsil/projects/isaid_imagery'



#####################
### FIND THE DATA ###
#####################

## what is the testing image path
test_img_path = project_path + '/data/testing/raw_images/'

## list the image files
img_files = os.listdir(test_img_path)


############################
### LOOP OVER THE IMAGES ###
############################

i= 26 #31 # 0
while i < len(img_files):

    ## grab the current file
    cur_img_file = img_files[i]
    img_path = test_img_path + cur_img_file

    ## load in the current image
    img = fn.load_image(img_path, odd_dims=True)

    
    #####################################
    ### LOOP OVER THE GAUSSIAN SIGMAS ###
    #####################################

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


            plt.ion()
            plt.imshow(labeled_img)

    








