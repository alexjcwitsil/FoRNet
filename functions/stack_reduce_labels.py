########################################
## ---- Stack then Reduce Labels ---- ##
########################################

import fornet as fn
import numpy as np

def stack_reduce_labels(labels, probabilities):

    ## loop over the list of labels (and probabilities)
    
    i=0
    while i < len(labels):

        ## grab the current label and probability and add a dimension
        cur_label = np.expand_dims(labels[i],0)
        cur_prob = np.expand_dims(probabilities[i],0)

        ################
        ### STACKING ###
        ################

        ## initilize the label and probability stack if they haven't yet 
        if i==0:
            label_stack = cur_label
            prob_stack = cur_prob
        elif i>0:
            label_stack = np.concatenate([label_stack, cur_label],axis=0)
            prob_stack = np.concatenate([label_stack, cur_label])


        i=i+1
        

    ################
    ### REDUCING ###
    ################

    ## figure out how to join the label and probability stack
    ##label_stack = np.random.randint(0,3,18).reshape(2,3,3)
    ##prob_stack = (np.random.randint(0,100,18)/100).reshape(2,3,3)

    ## concat each label and probability for each pixel in the original image
    label_prob_concat = np.concatenate([label_stack,prob_stack], axis=0)

    ## apply the find best label function to the correct axis
    seg_img = np.apply_along_axis(fn.find_best_lab, axis=0, arr=label_prob_concat)

    return(seg_img)



