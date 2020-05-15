########################################
## --- Generate Training Features --- ##
###                                  ###
## Read in raw and segmented data and ##
## extract features from each segement##
## or mask in each image.             ##
########################################

import os
import numpy as np
import pandas as pd
import cv2 
import json
from scipy.stats import kurtosis, skew
from matplotlib import pyplot as plt
import pickle
from matplotlib.path import Path

os.chdir('/home/alexwitsil/python_packages/imagefx')


############
## INPUTS ##
############

## We may use this on training or testing data
training_or_testing = 'testing'


###########################
### READ IN ANNOTATIONS ###
###########################

## read in the training information 
with open('./data/isaid/iSAID_train.json') as f:
    train_info = json.load(f)
    ## keys -> 'images', 'categories', 'annotations'


###################################
### GRAB ANNOTATION INFORMATION ###
###################################



## separate the annotations out
## this is a LIST with each list element being a dictionary
all_annotations = train_info['annotations']

## set up a list to hold the image ids
all_img_ids = np.zeros(len(all_annotations))

i=0
while i < len(all_annotations):
    #grab and save the image ID for this iteration 
    all_img_ids[i] = int(all_annotations[i]['image_id'])
    
    i=i+1


####################################
### FIND THE RAW TRAINING IMAGES ###
####################################

## list the dir and containing files
##raw_img_dir = './data/training/raw_images/'
raw_img_dir = './data/' + training_or_testing + '/raw_images/'
raw_img_files = os.listdir(raw_img_dir)


#######################
### LOOP RAW IMAGES ###
#######################

## how many images in the training data
num_images = len(train_info['images'])

j=0
while j<num_images :
    
    ## what are the names of the current image file (raw and instance)
    cur_img_raw_file = train_info['images'][j]['file_name']
    #cur_img_instance_file = train_info['images'][j]['ins_file_name']

    ## what is the current image ID
    cur_img_id = train_info['images'][j]['id']
    
    ## find where the values of the image ids that match the current image id
    cur_img_annotation_ids = np.where(all_img_ids == cur_img_id)

    ## check for raw image in current image directory.  If none, skip.
    if cur_img_raw_file not in raw_img_files:
        j=j+1
        continue
       
    ######################
    ### READ IN IMAGES ###
    ######################
    
    ## read in raw (grayscaled) image
    cur_img_raw = cv2.imread(raw_img_dir + cur_img_raw_file, cv2.IMREAD_GRAYSCALE)
    #cur_img_instances = cv2.imread(instance_masks_dir + cur_img_instance_file, cv2.IMREAD_GRAYSCALE)
    
    ## check to see if the image is odd or even... force it to be odd 
    # if(cur_img_raw.shape[0] % 2 == 0): 
    #     cur_img_raw = np.delete(cur_img_raw,-1,0)
    #     cur_img_instances = np.delete(cur_img_instances,-1,0)
        
    # if(cur_img_raw.shape[1] % 2 == 0):
    #     cur_img_raw = np.delete(cur_img_raw,-1,1)
    #     cur_img_instances = np.delete(cur_img_instances,-1,1)
        
    ## convert the image from int (I think) to a float
    cur_img = np.float64(cur_img_raw)

    ## initilize a background image (pixel values not planes, bridges, etc)
    bkg_img = np.copy(cur_img)
    
    
    ###############################
    ### FEATURE EXTRACTION LOOP ###
    ###############################
    
    ## what are the statistics you will extract?
    stat_names = ["sum", "mean", "sd", "skew", "kurt", "area", "category_id"]
    
    ## initilize a data frame to store cur img segmentation statistics 
    cur_img_seg_stats = pd.DataFrame(columns = stat_names)
    
    ## initilize a list to hold all the segmentation xy indicies
    all_seg_xys = list()
    
    ## loop over all the annotation indices
    i=0
    while i<len(cur_img_annotation_ids[0]):
        
        ## not sure why but this is a ndarray inside a tuple of length 1
        ## so index the tuple to get at the ndarray values 
        cur_annotation_ind = cur_img_annotation_ids[0][i] 
        
        
        ###############################
        ### GET THE CURRENT OUTLINE ###
        ###############################
        
        ## get the current annotation
        cur_annotation = train_info["annotations"][cur_annotation_ind] 
        ## keys -> 'id', 'image_id', 'segmentation', 'category_id', 'category_name', 'iscrowd', 'area', 'bbox'
        
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
        
        ####################################
        ### GET THE CURRENT SEGMENTATION ### 
        ####################################
        
        ## set up an array with outline xy values in columns
        outline_xys = np.column_stack((outline_xs,outline_ys))
        
        ## make a polygon for the current segmentation with matlibplot Path
        seg_polygon = Path(outline_xys,codes=None,closed=True)
        
        ## find the bounding box around the outline
        outline_left = np.min(outline_xs)
        outline_right = np.max(outline_xs)
        outline_bottom = np.min(outline_ys)
        outline_top = np.max(outline_ys)

        ## bounding x and y dimensions
        bounding_xdim = outline_right - outline_left
        bounding_ydim = outline_top - outline_bottom

        ## create an 0xN array that spans the bounding x and y dimensions
        bounding_xs = np.arange(bounding_xdim)
        bounding_ys = np.arange(bounding_ydim)

        ## create a mesh with these x and y bounding vectors
        bounding_mesh_xs,bounding_mesh_ys=np.meshgrid(bounding_xs,bounding_ys)

        ## flatten the bounding mesh grid into vectors and add the translation
        bounding_xs_vec = bounding_mesh_xs.flatten() + outline_left
        bounding_ys_vec = bounding_mesh_ys.flatten() + outline_bottom

        ## combine the bounding xy points into a 2 column array
        bounding_xys = np.vstack((bounding_xs_vec,bounding_ys_vec)).T

        ## which bounding xy indices are within the segmented polygon (bool)
        points_in_seg_vec = seg_polygon.contains_points(bounding_xys)

        ## convert this vector to a mesh (Bool)
        points_in_seg_mesh = points_in_seg_vec.reshape(bounding_ydim,bounding_xdim).transpose()

        ## find the segment xys that are within the bounding box region
        seg_xys_unadjusted = np.where(points_in_seg_mesh==True)

        ## separate and adjust the x and y locations
        seg_xs = seg_xys_unadjusted[0] + outline_left
        seg_ys = seg_xys_unadjusted[1] + outline_bottom

        ## create an (adjusted) segmentation x and y object
        seg_xys = (seg_xs, seg_ys)
        
        ## generate the segmented image as a binary and original valued img
        cur_seg_img = np.zeros(cur_img.size).reshape(cur_img.shape)
        cur_seg_bin = np.zeros(cur_img.size).reshape(cur_img.shape)
        
        ## populate the segemented image
        ## NOTE WE DONT USE THESE YET BUT THE YS AND XS ARE SWITCHED
        cur_seg_img[seg_ys, seg_xs] = cur_img[seg_ys, seg_xs]
        cur_seg_bin[seg_ys, seg_xs] = 1
        
        ## also save the current segementation values
        cur_seg_vals =  cur_img[seg_ys, seg_xs]
        
        ## update the background image
        bkg_img[seg_ys, seg_xs] = 0
        
        ##QUICK PLOTTING
        # plot_margin = 100
        # plt.ion()
        # plt.subplot(1,3,1)
        # plt.imshow(cur_img)
        # plt.xlim(outline_left-plot_margin,outline_right+plot_margin)
        # plt.ylim(outline_bottom-plot_margin,outline_top+plot_margin)

        # plt.subplot(1,3,2)
        # plt.imshow(cur_seg_img)
        # plt.xlim(outline_left-plot_margin,outline_right+plot_margin)
        # plt.ylim(outline_bottom-plot_margin,outline_top+plot_margin)

        # plt.subplot(1,3,3)
        # plt.imshow(cur_img)
        # plt.xlim(outline_left-plot_margin,outline_right+plot_margin)
        # plt.ylim(outline_bottom-plot_margin,outline_top+plot_margin)
        # plt.scatter(seg_xs, seg_ys)
        # ##plt.scatter(bounding_xs_vec, bounding_ys_vec)
        # ##plt.scatter(outline_xs, outline_ys)
        # move_on = input('iterate on? ')
        # ##plt.close('all')

        ##########################
        ### Extract STATISTICS ### 
        ##########################
        
        ## color features
        cur_sum = np.sum(cur_seg_vals)
        cur_mean = np.mean(cur_seg_vals)
        cur_sd = np.std(cur_seg_vals)
        cur_skew = skew(cur_seg_vals)
        cur_kurt = kurtosis(cur_seg_vals)
        
        ## spatial features
        cur_area = len(seg_xs)
        
        
        ######################################
        ### APPEND STATISTICS TO DATAFRAME ###
        ######################################
        
        ## save the current stats as a list 
        cur_stats = [cur_sum, cur_mean, cur_sd, cur_skew, cur_kurt, cur_area]
        
        ## add the current stats list to the current image segmentation statistics 
        cur_img_seg_stats.loc[i] = cur_stats + [cur_annotation_category_id]
        
        ## append the current segmentation xy inds to the all list
        all_seg_xys.append(seg_xys)
        
        print(cur_img_raw_file + ' segment ' + str(i) + ' out of: ' +  str(len(cur_img_annotation_ids[0])))
        
        i=i+1
    
    
    #########################################
    ### EXTRACT STATISTICS FROM BKG IMAGE ###
    #########################################
    
    ## find the x and y (segmentation)  values of the background image
    seg_xys = np.where(bkg_img != 0)
    seg_xs = seg_xys[0]
    seg_ys = seg_xys[1]
    
    ## also save the current segementation values
    cur_seg_vals =  bkg_img[seg_xs, seg_ys]
    
    ## color features
    cur_sum = np.sum(cur_seg_vals)
    cur_mean = np.mean(cur_seg_vals)
    cur_sd = np.std(cur_seg_vals)
    cur_skew = skew(cur_seg_vals)
    cur_kurt = kurtosis(cur_seg_vals)
    
    ## spatial features
    cur_area = len(seg_xs)
    
    ## save the current stats as a list 
    cur_stats = [cur_sum, cur_mean, cur_sd, cur_skew, cur_kurt, cur_area]
    
    ## define a background category
    cur_annotation_category = 'background'
    cur_annotation_category_id =  0 ## len(train_info['categories']) + 1 #16
    
    #add the current stats list to the current image segmentation statistics 
    cur_img_seg_stats.loc[i] = cur_stats + [cur_annotation_category_id]
    
    ## append the current segmentation xy inds to the all list
    all_seg_xys.append(seg_xys)
    
    
    ############
    ## SAVING ##
    ############
    
    # img_blob_info = [img.shape, all_blob_stats, blob_xy_inds]
    img_training_info = [cur_img.shape, cur_img_seg_stats, all_seg_xys]
    
    ## what is the file name to save
    ##out_file = './results/training_results/labeled_features/' + cur_img_raw_file[:-4] + '.pickle'
    out_file = './results/' + training_or_testing + '_results/labeled_features/' + cur_img_raw_file[:-4] + '.pickle'
    
    
    with open(out_file, 'w+b') as handle:
        pickle.dump(img_training_info, handle)
    
    print(j)
    j=j+1
   










