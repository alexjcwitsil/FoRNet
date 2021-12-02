################################
### NORMALIZE IMAGE CHANNELS ###
################################

import numpy as np

def norm_chans(img):
    num_chans = img.shape[2]
    out_img = img.copy()
    i=0
    while i < num_chans:
        out_img[:,:,i] = (img[:,:,i]-np.mean(img[:,:,i]))/np.std(img[:,:,i])
        #out_img[:,:,i] = np.abs(out_img[:,:,i])
        
        i=i+1
    #
    return(out_img)


