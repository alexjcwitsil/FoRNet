######################################
## ---- Find Points in Polygon ---- ##
###                                ###
## Given an outline of x and y values#
## find all the x and y values within#
## that polygon/outline.            ##
######################################

from matplotlib.path import Path
import numpy as np

def points_in_polygon(xy_array):

    ## make a polygon for the current segmentation with matlibplot Path
    seg_polygon = Path(xy_array,codes=None,closed=True)

    ## break the xy array into x and y values
    outline_xs = xy_array[:,0]
    outline_ys = xy_array[:,1]

    ## find the bounding box around the outline
    outline_left = np.min(outline_xs)
    outline_right = np.max(outline_xs)
    outline_bottom = np.min(outline_ys)
    outline_top = np.max(outline_ys)

    ## bounding x and y dimensions
    bounding_xdim = outline_right - outline_left
    bounding_ydim = outline_top - outline_bottom

    ## create an 0xN array that spans the bounding x and y dimensions
    bounding_xs = np.arange(bounding_xdim)
    bounding_ys = np.arange(bounding_ydim)

    ## create a mesh with these x and y bounding vectors
    bounding_mesh_xs,bounding_mesh_ys=np.meshgrid(bounding_xs,bounding_ys)

    ## flatten the bounding mesh grid into vectors and add the translation
    bounding_xs_vec = bounding_mesh_xs.flatten() + outline_left
    bounding_ys_vec = bounding_mesh_ys.flatten() + outline_bottom

    ## combine the bounding xy points into a 2 column array
    bounding_xys = np.vstack((bounding_xs_vec,bounding_ys_vec)).T

    ## which bounding xy indices are within the segmented polygon (bool)
    points_in_poly_vec = seg_polygon.contains_points(bounding_xys)

    ## convert this vector to a mesh (Bool)
    points_in_poly_mesh = points_in_poly_vec.reshape(bounding_ydim,bounding_xdim).transpose()

    ## find the xys indices that are within the bounding box region
    xy_inds_unadjusted = np.where(points_in_poly_mesh==True)

    ## separate and adjust the x and y locations
    x_inds = xy_inds_unadjusted[0] + outline_left
    y_inds = xy_inds_unadjusted[1] + outline_bottom

    ## create an (adjusted) segmentation x and y object
    xy_inds = (x_inds, y_inds)

    return(xy_inds)

