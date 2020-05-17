##########################################
## ----          TRAINING          ---- ##
###                                    ###
##########################################

import fornet as fn
import json
import pickle
import os 

def train(project_path, gaus_sigs):

    ##############
    ### INPUTS ###
    ##############

    ## Gaussian in the LoG algorithm
    #gaus_sigs = [8, 10]
    
    ## where is the project
    #project_path = '/home/alexwitsil/projects/isaid_imagery'

    # what is the raw image directory
    raw_img_path = project_path + '/data/training/raw_images/'

    # list the image files
    img_files = os.listdir(raw_img_path)


    ####################################
    ### READ IN TRAINING ANNOTATIONS ###
    ####################################

    ## read in the training information 
    with open(project_path + '/data/iSAID_train.json') as f:
        train_info = json.load(f)
        ## keys -> 'images', 'categories', 'annotations'


    ###########################
    ## PARSE ANNOTATION INFO ##
    ###########################

    parsed_training_info = fn.parse_training_info(train_info)


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

        j = 40 #0 #27 
        while j < len(img_files):

            ###########################
            ## LOAD IN CURRENT IMAGE ##
            ###########################

            ## define the path to the current image
            cur_img_file = img_files[j]
            img_path = raw_img_path + cur_img_file

            ## load in the current image
            img = fn.load_image(img_path, odd_dims=True)


            ############################
            ## GENERATE BLOB FEATURES ##
            ############################

            img_blob_info = fn.gen_blob_features(img, cur_sig)


            #################################
            ## GENERATE SEGMENTED FEATURES ##
            #################################

            img_seg_info = fn.gen_segmented_features(img, cur_img_file, train_info, parsed_training_info)


            ##############################
            ## GENERATE SEGMENTED IMAGE ##
            ##############################

            img_true_seg = fn.gen_segmented_image(img_seg_info)


            ##################################
            ## GENERATE BACKGROUND FEATURES ##
            ##################################

            img_bkg_info = fn.gen_background_features(img_blob_info, img_true_seg)


            #################################
            ## JOIN SEGMENTED AND BKG INFO ##
            #################################

            img_features_unnorm = fn.join_seg_bkg_features(img_seg_info, img_bkg_info)

            ########################################
            ## ADD IMAGE FEATURES TO ALL FEATURES ##
            ########################################

            if j == 0:
                ## initilize the pandas dataframe
                all_features = img_features_unnorm
            if j > 0:
                all_features =all_features.append(img_features_unnorm)

            j=j+1


        ###############
        ### RUN ANN ###
        ###############

        model_results = train_vanilla_ann(all_features,num_epochs=5)

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
            pickle.dump(sc, handle)

        # save model and architecture to single file
        model.save(save_dir + model_file)


        i=i+1

    
























