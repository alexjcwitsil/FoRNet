import math
import numpy as np

def build_gaus2d(xdim, ydim, sig, cent=[]):

    if isinstance(sig, int):
        sig = [sig, sig]

    if len(cent) == 0:
        cent = [math.ceil(xdim/2)-1, math.ceil(ydim/2)-1]

    ## separate the center
    cent_x = cent[0]
    cent_y = cent[1]

    sig_y = sig[0]
    sig_x = sig[1]

    ## build a mesh with dimensions xdim x ydim
    mesh = np.array(np.meshgrid(np.arange(xdim), np.arange(ydim)))

    ## flatten the mesh to get vector of x and y values
    xs = (mesh[0].flatten())
    ys = (mesh[1].flatten())

    ## calculate the gaussian unnormalized value at each x and y value
    gaus_vec_unnorm = np.exp(-((xs-cent_x)**2/(2*sig_x**2)+(ys-cent_y)**2/(2*sig_y**2)))

    ## normalize the gaussian values
    gaus_vec = gaus_vec_unnorm / np.sum(gaus_vec_unnorm)

    ## not sure why you need to reshape with ydim THEN xdim and then transpose?
    gaus2d = gaus_vec.reshape(ydim, xdim).transpose()

    ##plt.imshow(gaus2d)
    
    return(gaus2d)





