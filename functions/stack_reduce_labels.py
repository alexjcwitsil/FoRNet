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


            ## prob_stack = np.concatenate([label_stack, cur_label])!!!!!
            prob_stack = np.concatenate([prob_stack, cur_prob])

            

        i=i+1
        

    ################
    ### REDUCING ###
    ################

    unique_labels = np.unique(label_stack)

    ## initilize an array to hold the weighted labels
    weighted_labels = np.zeros((len(unique_labels),labels[0].shape[0], labels[0].shape[1]))


    ## loop over all the labels
    i=0
    while i < len(unique_labels):

        ## what is the current label
        cur_lab = unique_labels[i]

        ## isolate the current label in the label_stack
        cur_label_stack = np.zeros(label_stack.shape)
        cur_prob_stack = np.copy(prob_stack)

        cur_label_stack[label_stack == cur_lab] = 1
        cur_prob_stack[label_stack != cur_lab] = 0

        cur_weighted_stack = cur_label_stack * cur_prob_stack
        
        ## stack according to the mean, median, or max. 
        ##cur_weighted_label = np.median(cur_weighted_stack, axis=0)
        ##cur_weighted_label = np.max(cur_weighted_stack, axis=0)
        cur_weighted_label = np.mean(cur_weighted_stack, axis=0)

        ## save the current weighted label
        weighted_labels[i,:,:] = cur_weighted_label

        i=i+1
    #

    ## find out the heightest weighted label
    label_inds = np.argmax(weighted_labels, axis=0)

    ## index the unique labels to find the segmented image
    seg_img_list = [unique_labels[i] for i in label_inds.flatten()]
    seg_img = np.array(seg_img_list).reshape(labels[0].shape)
    

    # ## figure out how to join the label and probability stack
    # ##label_stack = np.random.randint(0,3,18).reshape(2,3,3)
    # ##prob_stack = (np.random.randint(0,100,18)/100).reshape(2,3,3)

    # ## concat each label and probability for each pixel in the original image
    # label_prob_concat = np.concatenate([label_stack,prob_stack], axis=0)

    # ## apply the find best label function to the correct axis
    # seg_img = np.apply_along_axis(fn.find_best_lab, axis=0, arr=label_prob_concat)

    return(seg_img)



