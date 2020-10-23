##########################################
## ----          TRAINING          ---- ##
###                                    ###
##########################################

import fornet as fn
import json
import pickle
import numpy as np
import pandas as pd
import os 

def train(project_path, gaus_sigs, bkg_ignore=False):

    ##############
    ### INPUTS ###
    ##############

    ## setup/check the testing and training data dirs
    fn.split_train_test(project_path + '/data/', test_size=0.2)

    # what is the raw image directory
    raw_img_path = project_path + '/data/training/raw_images/'

    # list the image files
    img_files = os.listdir(raw_img_path)


    #################################
    ### READ IN IMAGE ANNOTATIONS ###
    #################################

    ## find the json file (if not provided...)
    label_info_file = [f for f in os.listdir(project_path + '/data/') if f.endswith('.json')][0]
    with open(project_path + '/data/' + label_info_file) as f:
        label_info = json.load(f)


    ###########################
    ## PARSE ANNOTATION INFO ##
    ###########################

    parsed_label_info = fn.parse_label_info(label_info)


    #####################################
    ### LOOP OVER THE GAUSSIAN SIGMAS ### 
    #####################################

    i=0
    while i < len(gaus_sigs):

        ## grab the current Gaussian sigma
        cur_sig = gaus_sigs[i]


        #########################
        ### SETUP RESULT DIRS ###
        #########################

        ## set up the results directories for this project
        fn.setup_result_dirs(cur_sig, project_path)


        ###########################
        ## LOOP OVER IMAGE FILES ##
        ###########################

        j = 0 
        while j < len(img_files):

            ###########################
            ## LOAD IN CURRENT IMAGE ##
            ###########################

            ## define the path to the current image
            cur_img_file = img_files[j]
            img_path = raw_img_path + cur_img_file

            ## load in the current image
            img = fn.load_image(img_path, gray=False, odd_dims=True)
            
            ############################
            ## GENERATE BLOB FEATURES ##
            ############################

            img_blob_info = fn.gen_blob_features(img, cur_sig)


            #################################
            ## GENERATE SEGMENTED FEATURES ##
            #################################

            img_seg_info = fn.gen_segmented_features(img, cur_img_file, label_info, parsed_label_info)


            ##############################
            ## GENERATE SEGMENTED IMAGE ##
            ##############################

            img_true_seg = fn.gen_segmented_image(img_seg_info)


            ##############################
            ## GENERATE INNER  FEATURES ##
            ##############################

            img_inner_info = fn.gen_inner_features(img_blob_info, img_true_seg)


            #################################
            ## JOIN SEGMENTED AND BKG INFO ##
            #################################

            ##img_features_unnorm = fn.join_seg_bkg_features(img_seg_info, img_bkg_info)
            img_features_unnorm = fn.join_seg_inner_features(img_seg_info, img_inner_info)

            ########################################
            ## ADD IMAGE FEATURES TO ALL FEATURES ##
            ########################################

            if j == 0:
                ## initilize the pandas dataframe
                all_features = img_features_unnorm
            if j > 0:
                all_features =all_features.append(img_features_unnorm)

            print('image: ' + str(j+1) + ' out of: ' + str(len(img_files)))
            j=j+1


        ###############
        ### RUN ANN ###
        ###############

        ## if you want to ignore the background, remove assocaited rows
        if bkg_ignore == True:
            ## remove those rows from the all_features dataframe
            all_features = all_features[all_features['category_id'] != 0]
        #
        

        model_results = fn.train_vanilla_ann(all_features,num_epochs=1000)

        model = model_results[0]
        scaling_factors = model_results[1]
        model_history = model_results[2]



        ##############
        ### SAVING ###
        ##############


        #######################
        ## SAVE THE FEATURES ##
        #######################

        feature_save_path = '/results/training_results/'
        feature_save_file = 'sig' + str(cur_sig) + '_training_features.pickle'
        feature_save_loc = project_path+feature_save_path + feature_save_file
        all_features.to_pickle(feature_save_loc)


        ########################
        ## SAVE THE ANN MODEL ##
        ########################

        ## saving dir
        save_dir = project_path+'/results/nn_models/sig'+str(cur_sig)+'/'

        ## scaling factors and model file name
        scaling_factors_file = 'scaling_factors.pickle'
        model_file = 'vanilla_model.h5'

        ## save the scaling factors in the data
        with open(save_dir + scaling_factors_file,'w+b') as handle:
            pickle.dump(scaling_factors, handle)

        # save model and architecture to single file
        model.save(save_dir + model_file)


        i=i+1

    
























