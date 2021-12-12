import fornet as fn


def gen_for(img, sig, channel, img_file, label_info, parsed_label_info=None, img_meta=None):


    ###########################
    ## PARSE ANNOTATION INFO ##
    ###########################

    if parsed_label_info == None: 
        parsed_label_info = fn.parse_label_info(label_info)
    #


    ############################
    ## GENERATE BLOB FEATURES ##
    ############################

    ## Get blobs associated with the current channel. 
    img_blob_info = fn.gen_blob_features(img, sig, chan=channel, img_meta=img_meta)


    #################################
    ## GENERATE SEGMENTED FEATURES ##
    #################################

    img_seg_info = fn.gen_segmented_features(img, img_file, label_info, parsed_label_info, img_meta)


    ##############################
    ## GENERATE SEGMENTED IMAGE ##
    ##############################

    img_true_seg = fn.gen_segmented_image(img_seg_info)


    #############################
    ## GENERATE INNER FEATURES ##
    #############################

    img_inner_info = fn.gen_inner_features(img_blob_info, img_true_seg)

    #################################
    ## JOIN SEGMENTED AND BKG INFO ##
    #################################
    
    img_features_unnorm = fn.join_seg_inner_features(img_seg_info, img_inner_info)



    return(img_features_unnorm)



