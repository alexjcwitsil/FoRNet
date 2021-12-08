#############################
### RANGE BETWEEN 0 and 1 ###
#############################

import numpy as np

def range01(x):
    x_out = (x-np.min(x)) / (np.max(x) - np.min(x))

    ## replace any nans with 0
    ## this can occur when an array is small (len<=2) or when the minimum equals the maximum
    x_out[np.isnan(x_out)] = 0

    return(x_out)

    






