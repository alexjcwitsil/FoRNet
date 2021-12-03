#############################
### RANGE BETWEEN 0 and 1 ###
#############################

import numpy as np

def range01(x):
    x_out = (x-np.min(x)) / (np.max(x) - np.min(x))
    return(x_out)

    






