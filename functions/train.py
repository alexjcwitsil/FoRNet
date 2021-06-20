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
    fn.split_train_test(project_path + '/data/', test_size=0.0)

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


        ####################################
        ### LOOP OVER THE IMAGE CHANNELS ###
        ####################################

        ## read in a dummy image to see how many channels there are
        dme_img = fn.load_image(raw_img_path + img_files[0], gray=False)

        ## how many channels are there
        num_chans = dme_img.shape[2]

        j=0
        while j < num_chans:

            ## define the current channel
            cur_chan = j


            ###########################
            ## LOOP OVER IMAGE FILES ##
            ###########################

            k = 0 
            while k < len(img_files):
                
                ###########################
                ## LOAD IN CURRENT IMAGE ##
                ###########################

                ## define the path to the current image
                cur_img_file = img_files[k]
                img_path = raw_img_path + cur_img_file

                ## load in the current image
                img = fn.load_image(img_path, gray=False, odd_dims=True)


                ############################
                ## GENERATE BLOB FEATURES ##
                ############################

                ## Get blobs associated with the current channel. 
                img_blob_info = fn.gen_blob_features(img, cur_sig, chan=cur_chan)


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

                if k == 0:
                    ## initilize the pandas dataframe
                    all_features = img_features_unnorm
                if k > 0:
                    all_features =all_features.append(img_features_unnorm)

                print('image: ' + str(k+1) + ' out of: ' + str(len(img_files)))
                k=k+1
            #

            ###############
            ### RUN ANN ###
            ###############

            ## if you want to ignore the background, remove assocaited rows
            if bkg_ignore == True:
                ## remove those rows from the all_features dataframe
                all_features = all_features[all_features['category_id'] != 0]
            #

            ## determine the number of classes
            num_classes = int(len(label_info['categories'])) + 1 # for background

            ## adjust if ignoring background
            ##if bkg_ignore == True: num_classes = num_classes - 1

            ## train the model
            model_results = fn.train_vanilla_ann(all_features,num_epochs=500, num_classes=num_classes)

            model = model_results[0]
            scaling_factors = model_results[1]
            model_history = model_results[2]



            ##############
            ### SAVING ###
            ##############


            #######################
            ## SAVE THE FEATURES ##
            #######################

            feature_save_path = 'results/training_results/'
            feature_save_file = 'chan' + str(cur_chan) + '_sig' + str(cur_sig) + '_training_features.pickle'
            feature_save_loc = project_path+feature_save_path + feature_save_file
            all_features.to_pickle(feature_save_loc)


            ########################
            ## SAVE THE ANN MODEL ##
            ########################

            ## saving dir
            save_dir = project_path+'/results/nn_models/sig'+str(cur_sig)+'/'

            ## scaling factors and model file name
            scaling_factors_file = 'chan'+str(cur_chan)+'_scaling_factors.pickle'
            model_file = 'chan' + str(cur_chan) + '_vanilla_model.h5'

            ## save the scaling factors in the data
            with open(save_dir + scaling_factors_file,'w+b') as handle:
                pickle.dump(scaling_factors, handle)

            # save model and architecture to single file
            model.save(save_dir + model_file)

            j=j+1 ## move to the next image channel
        #

        i=i+1 ## move to the next Gaussian Sigma.

    
























