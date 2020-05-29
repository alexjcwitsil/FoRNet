###########################################
## ---- Evaluate Image Segmenations ---- ##
###                                     ###
###########################################

import fornet as fn
import numpy as np


def eval_segs(true_img, pred_img):

    
    ## FORCE IMAGES TO HAVE SAME SHAPE
    ## recall predicted images are forced to have odd dimensions
    ## therefore we need to force same constraint on true images
    if(true_img.shape[0] % 2 == 0): 
        true_img = np.delete(true_img,-1,0)
    if(true_img.shape[1] % 2 == 0):
        true_img = np.delete(true_img,-1,1)


    ## what are the unique labels/classes in each image
    unique_pred = np.unique(pred_img)
    unique_true = np.unique(true_img)

    ## setup a an array the ious for each image
    ious = np.zeros(len(unique_pred))


    ##################################
    ### LOOP OVER PREDICTED LABELS ###
    ##################################
    
    ## loop over every predicted segment label
    j=0
    while j < len(unique_pred):

        ## what is the current predicted label
        cur_pred_label = unique_pred[j]

        ## find where the predicted image matches the current predicted label
        cur_pred_label_xys = np.where(pred_img == cur_pred_label)

        ## binarize the predicted image such that only the current label is 1
        pred_bin = pred_img * 0
        pred_bin[cur_pred_label_xys[0],cur_pred_label_xys[1]] = 1

        cur_true_label = cur_pred_label

        ## binarize the true image such that only the current label is 1
        ## and the rest is 0
        true_bin = true_img * 0
        cur_true_label_xys = np.where(true_img == cur_true_label)
        true_bin[cur_true_label_xys[0],cur_true_label_xys[1]] = 1


        #####################
        ### CALCULATE IOU ###
        #####################

        ## first add the true and pred binary images together
        stacked_img = pred_bin + true_bin

        ## calculate area of overlap
        area_of_overlap = len(np.where(true_bin + pred_bin == 2)[0])

        ## true and predited areas that don't overlap
        pred_isolated_area = len(np.where(stacked_img == 1)[0])
        true_isolated_area = len(np.where(stacked_img == 1)[0])

        ## calculate area of 
        area_of_union = area_of_overlap + pred_isolated_area + true_isolated_area

        ## calculate IoU
        cur_iou = area_of_overlap / area_of_union


        ## add the current iou to all the ious
        ious[j] = cur_iou

        
        j=j+1
    
    return(ious)

