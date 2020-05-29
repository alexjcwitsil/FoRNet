
## dependencies
import numpy as np

def find_best_lab(label_prob_vec):
    ## break the vector into individual labels and probabilities
    lab_vec = label_prob_vec[0:int(len(label_prob_vec)/2)].astype(int)
    prob_vec = label_prob_vec[int(len(label_prob_vec)/2):]
    
    ## find all the unique labels in the label vector
    unique_labs = np.unique(lab_vec)
    
    ## initilize an array to hold all the label probabilities
    all_lab_probs = np.zeros(len(unique_labs) * 2).reshape(len(unique_labs),2)
    all_lab_probs[:,0] = unique_labs
    
    ## loop over all the unique labels 
    i=0
    while i < len(unique_labs):
        cur_lab = unique_labs[i]
        
        ## where are the current labels
        cur_lab_inds = np.where(lab_vec == cur_lab)
        
        ## take the sum of the probabilities at these indices
        cur_lab_probs = np.array(prob_vec)[cur_lab_inds[0]]
        
        ## save the cur label and its summed probability 
        all_lab_probs[i,1] = np.sum(cur_lab_probs)
        
        i=i+1
    
    ## find the label with the highest (best) probability
    best_lab_ind = np.where(all_lab_probs[:,1] == np.max(all_lab_probs[:,1]))

    ## there could be multiple labels with the same probability.
    ## take the first one...for now...
    best_lab_ind = best_lab_ind[0][0]
    best_lab = int(all_lab_probs[best_lab_ind,0])
    
    return(best_lab)



