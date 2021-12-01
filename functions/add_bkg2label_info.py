############################################
## ---- Add Background to Label Info ---- ##
############################################


def add_bkg2label_info(label_info):

    ii = 0
    while ii < len(label_info['categories']):

        ## grab the current category from the label info
        cur_cat = label_info['categories'][ii]

        ## grab the current category ID and add 1 to it 
        cur_cat_id = cur_cat['id']
        new_cat_id = cur_cat_id + 1

        ## replace the category id with the new category id
        label_info['categories'][ii]['id'] = new_cat_id

        ii=ii+1
    #

    ## loop over the annotation keys
    ii=0
    while ii < len(label_info['annotations']):

        ## grab the current annotation
        cur_annotation = label_info['annotations'][ii]

        ## grab the curren annotation category id and add 1 to it
        cur_annotation_cat_id = cur_annotation['category_id'] 
        new_annotation_cat_id = cur_annotation_cat_id + 1

        ## replace the annotation category id with the new one
        label_info['annotations'][ii]['category_id'] = new_annotation_cat_id

        ii=ii+1
    #

    return(label_info)
#
    







