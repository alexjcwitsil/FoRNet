#############################################
### Normalize IMG Channel Between 0 and 1 ###
#############################################

import fornet as fn

def img_range01(img, max=1):

    num_chans = img.shape[2]
    img01 = img.copy()

    i=0
    while i < num_chans:
        img01[:,:,i] = fn.range01(img[:,:,i])
        img01[:,:,i] = img01[:,:,i]*max
        i=i+1
    #

    return(img01)


