################################
## --- Grayscale an Image --- ##
################################

import numpy as np

def grayscale_img(img):

    ## check if we need to grayscale the image
    if(len(img.shape) == 2):

        ## the image is already grayscaled
        img_gray = img
        
    elif(len(img.shape) == 3):

        ## initilize a grayscaled image
        img_gray = np.zeros(img[:,:,0].size).reshape(img.shape[0:2])

        ## take the average across all the color channels
        i=0
        while i < img.shape[2] :

            ## what is the current channel
            cur_chan = img[:,:,i]

            ## add the curren channel to the grayscaled image
            img_gray = img_gray + cur_chan

            i=i+1
        #

        ## take the average of the summed grayscaled image
        img_gray = img_gray / img.shape[2]

    #
    
    ## return the grayscaled image
    return(img_gray)

