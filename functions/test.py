
import fornet as fn
import os
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model


def test(project_path, gaus_sigs,bkg_ignore=False):

     
    #################################
    ### READ IN IMAGE ANNOTATIONS ###
    #################################

    ## read in the training information 
    with open(project_path + '/data/label_info.json') as f:
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
        img = fn.load_image(img_path, gray=False, odd_dims=True)


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
            ##cur_features = sc.fit_transform(blob_features_unnorm)
            cur_features = sc.transform(blob_features_unnorm)
            

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
            #

            ## if you ignored backgroudn, you must add 1 to the predictions
            if bkg_ignore == True:
                pred = [i+1 for i in pred]
            #
            

            
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


        ##############
        ### SAVING ###
        ##############

        ## true image and predicted image directories
        true_img_dir = project_path + 'results/testing_results/true_image_segmentations/'
        pred_img_dir = project_path + 'results/testing_results/predictions/stacked/labeled_images/'

        ## save the true image segmentation
        np.save(true_img_dir + cur_img_file[:-4] + '.npy', true_img_seg)
        np.save(pred_img_dir + cur_img_file[:-4] + '.npy', pred_img_seg)


        print('')
        print(f' Testing and evaluating file {str(i)} ({img_files[i]}) out of {str(len(img_files))}\r', end="")
        i=i+1
    print()

    
    ################
    ### EVALUATE ###
    ################

    ## where are the predicted and true segmentation directories
    true_img_dir = project_path + 'results/testing_results/true_image_segmentations/'
    pred_img_dir = project_path + 'results/testing_results/predictions/stacked/labeled_images/'

    ## list the files
    true_img_files = os.listdir(true_img_dir)
    pred_img_files = os.listdir(pred_img_dir)


    ## how many label categories are there
    num_cats = len(label_info['categories']) + 1 # add one for background!

    ## how many test images are there
    num_imgs = len(true_img_files)

    ## set up a pandas dataframe to hold the IOUS for each test image
    all_ious = np.zeros(num_cats * num_imgs).reshape(num_imgs, num_cats)
    all_ious[:] = np.nan

    ## loop over all image files
    i=0
    while i < len(true_img_files): 

        true_img = np.load(true_img_dir + true_img_files[i])
        pred_img = np.load(pred_img_dir + pred_img_files[i])

        ## calculate the intersection over union
        img_ious = fn.eval_segs(true_img, pred_img,bkg_ignore)

        ## set up an array of ious if first iteration
        ##if i == 0:
            ## set up an array to hold all the ious
            ##all_ious = np.zeros(len(true_img_files)*len(img_ious)).reshape(len(true_img_files), len(img_ious))

        ##all_ious[i,:] = img_ious
        ## populate the ious
        j=0
        while j < img_ious.shape[0]:
            all_ious[i, int(img_ious[j,0])] = img_ious[j,1]
            j=j+1
        #
        

        i=i+1
    #
    ## make a data frame with the ious and include the image name
    iou_df = pd.DataFrame(all_ious)

    ## add the label names to the dataframe columns
    col_names = [i['name'] for i in label_info['categories']]
    iou_df.columns = ['background'] + col_names

    iou_df['images'] = img_files



    ## save the IOUs
    iou_df.to_pickle(project_path + 'results/testing_results/image_ious.pickle')
    np.save(project_path + 'results/testing_results/image_ious.npy', all_ious)

    
    return(iou_df)



