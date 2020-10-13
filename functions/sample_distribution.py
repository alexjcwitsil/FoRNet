#####################################################
## ---- Sample Data from a given Distribution ---- ##
#####################################################

import numpy as np
from matplotlib import pyplot as plt

def sample_distribution(data, n_samps, distribution = 'flat'):

    ## initilize a reordered dataset
    data_reordered = np.zeros(len(data))


    if distribution == 'flat':
        ## find min and max of data
        min_ind = 0
        max_ind = len(data) - 1

        ## create the sample indices 
        sample_inds = np.linspace(min_ind,max_ind,n_samps).astype(int)

        ## use the indices to sample the data
        data_sampled = [data[i] for i in sample_inds]
    #

    if distribution == 'left':
        data_sampled = data[0:n_samps]
    #  

    if distribution == 'right':
        data_sampled = data[-n_samps:]
    #


    if distribution == 'bimodal':
        data_dummy = data[:]
        i=0
        while len(data_dummy) > 0:
            ## find the first and second sampled index
            cur_samp_ind1 = 0
            cur_samp_ind2 = len(data_dummy)-1

            ## need to consider if there odd or even number of data entries
            if i == (len(data) - 1):
                ## add the current sample to the sampled data list
                data_reordered[i] = data_dummy[cur_samp_ind1]
            
                ## combine the indices
                all_cur_samp_inds = [cur_samp_ind1]
            #
            if i < (len(data) - 1): 
                ## add the current sample to the sampled data list
                data_reordered[i] = data_dummy[cur_samp_ind1]
                data_reordered[i+1] = data_dummy[cur_samp_ind2]

                ## combine the indices
                all_cur_samp_inds = [cur_samp_ind1, cur_samp_ind2]
            #

            ## find the current values from the dummy list
            cur_vals = [data_dummy[i] for i in all_cur_samp_inds]
        
            ## remove the curretn samples from the dummy list
            data_dummy = [i for i in data_dummy if i not in cur_vals]

            i=i+2
        #

        ## get the number of required samples from the current values
        data_sampled = data_reordered[0:n_samps]
    #


    if distribution == 'trimodal':
        data_dummy = data[:]
        i=0
        while len(data_dummy) > 0:

            if i == (len(data) - 1):
                cur_samp_ind1 = int(np.floor(len(data_dummy)/2))

                ## add the current sample to the sampled data list
                data_reordered[i] = data_dummy[cur_samp_ind1]
                all_cur_samp_inds = [cur_samp_ind1]
                #
            if i == (len(data) - 2):
                cur_samp_ind1 = 0
                cur_samp_ind2 = len(data_dummy) - 1

                ## add the current sample to the sampled data list
                data_reordered[i] = data_dummy[cur_samp_ind1]
                data_reordered[i+1] = data_dummy[cur_samp_ind2]

                ## combine the indices
                all_cur_samp_inds = [cur_samp_ind1, cur_samp_ind2]
            #
            if i < (len(data) - 2):
                cur_samp_ind1 = 0
                cur_samp_ind2 = int(np.floor(len(data_dummy)/2))
                cur_samp_ind3 = len(data_dummy)-1

                ## add the current sample to the sampled data list
                data_reordered[i] = data_dummy[cur_samp_ind1]
                data_reordered[i+1] = data_dummy[cur_samp_ind2]
                data_reordered[i+2] = data_dummy[cur_samp_ind3]

                ## combine the indices
                all_cur_samp_inds=[cur_samp_ind1, cur_samp_ind2,cur_samp_ind3]
            #

            cur_vals = [data_dummy[i] for i in all_cur_samp_inds]

            ## remove teh current samples from the dummy list
            data_dummy = [i for i in data_dummy if i not in cur_vals]

            i=i+3
        #

        ## grab the number of required number of samples from the current values
        data_sampled = data_reordered[0:n_samps]
    #


    if distribution == 'normal':
        data_dummy = data[:]
        i=0
        while len(data_dummy) > 0:

            ## find the current sample index
            cur_samp_ind = int(np.floor(len(data_dummy)/2))

            ## add the current sample the data sampled list
            data_reordered[i] = data_dummy[cur_samp_ind]

            ## remove the current sample index from the dummy list
            data_dummy.pop(cur_samp_ind)

            i=i+1
        #

        ## grab the number of required number of samples from the current values
        data_sampled = data_reordered[0:n_samps]
    #

    return(list(data_sampled))


