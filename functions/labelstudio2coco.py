
import fornet as fn
import json
import os


def labelstudio2coco(completion_path, img_path):


    ### DME IF WORKING ###
    ## this is oncly to compare true coco format iwth labelstudio converted format
    ## use these coco data to structure the conversion algorithm
    ##coco_compare_path = '/home/alexwitsil/projects/isaid_imagery'

    

    ### DME IF WORKING ###
    ## these are now inputs
    ## project, completions, and image files
    # project_path = '/home/alexwitsil/projects/galapagos_image_classification/'
    # completion_path = project_path + '/image_labeling/test_labeling/completions/'
    # img_path = project_path + './sandbox/resized_images'

    img_files = os.listdir(img_path)


    ## initlize properly formated coco object
    coco = {}
    coco['images'] = []
    coco['categories'] = []
    coco['annotations'] = []

    ## initilize a list to hold all the categories
    all_categories = [{'id':'dme', 'name':'dme'}]


    ### DME IF WORKING ###
    ## this is only to compare true coco format with labelstudio converted format
    ## read in the training information 
    ## with open(coco_compare_path + '/data/iSAID_train.json') as f:
        ## label_info = json.load(f)
        ## keys -> 'images', 'categories', 'annotations'


    ## list all the completion files
    completion_files = os.listdir(completion_path)

    ## LOOP OVER ALL COMPLETION FILES 
    j=0
    while j < len(completion_files):
 
        ## read in the labelstudio information
        with open (completion_path + completion_files[j]) as f:
            cur_completion = json.load(f)

        ## find the name of the current image used for labeling
        cur_img_file = [i for i in img_files if i in cur_completion['data']['image']][0]

        ## read in the image just to check polygons
        ##img = fn.load_image(project_path + '/sandbox/resized_images/GAL010_00037.JPG')

        ## parse the cur_completion labelstudio format into elements
        cur_completion_results = cur_completion['completions'][0]['result']


        ## Grab the current image id
        cur_image_id = j
        cur_ins_file_name = ''
        cur_seg_file_name = ''
        cur_seg_file_name = ''

        ## add the image info the coco format
        coco['images'].append({'id': cur_image_id, 'file_name': cur_img_file, 'ins_file_name': cur_ins_file_name, 'seg_file_name': cur_seg_file_name})


        ## loop over the cur_completion results
        i=0
        while i < len(cur_completion_results):

            ## grab the current result
            cur_result = cur_completion_results[i]

            ## what is the image shape
            img_shape = [cur_result['original_width'], cur_result['original_height']]

            ## force image shape to be odd
            ## NOTE if segmentations extend to boundary of image with EVEN
            ## dimensions, extracting features will fail bc 'out of bounds'
            if img_shape[0] % 2 == 0:
                img_shape[0] = img_shape[0] - 1
            if img_shape[1] % 2 == 0:
                img_shape[1] = img_shape[1] - 1 


            #################
            ## GRAB THE ID ##
            #################
            ## this is an iterator that accounts for all annotations in all the images (completions)
            cur_id = len(coco['annotations']) #i + j


            ###########################
            ## GRAB THE SEGMENTATION ##
            ###########################

            ## get the points from the current result
            cur_completion_points = cur_result['value']['points']

            ## get the x and y points (initially they are percents) and round them
            xs_perc = [round(x[0],0) for x in cur_completion_points]
            ys_perc = [round(x[1],0) for x in cur_completion_points]

            ## convert those percents into integer points according to image dimensions
            xs = [int(i * img_shape[0]*0.01) for i in xs_perc]
            ys = [int(i * img_shape[1]*0.01) for i in ys_perc]

            ## zip the xs and ys together
            xys_zip = list(zip(xs,ys))

            ## get rid of the of the embeded tuple
            xys = [i for j in xys_zip for i in j]

            ## name these segmentation according to coco format
            cur_segmentation = xys

            ############################
            ## GRAB THE CATEGORY NAME ## 
            ############################

            cur_category_name = cur_result['value']['polygonlabels'][0]


            ##########################
            ## GRAB THE CATEGORY ID ##
            ##########################

            ## check if there is already an instance of the current category
            cur_category_id_check = [i for i in all_categories if i['name'] == cur_category_name]

            ## add the current category name with appropriate id if it doesn't already exist
            if len(cur_category_id_check) == 0:
                ## if there is no instnace of the current category create an id number
                cur_category_id = len(all_categories)
                all_categories.append({'id': cur_category_id, 'name': cur_category_name})
            elif len(cur_category_id_check) > 0:
                ## grab the id since it already exists
                cur_category_id = cur_category_id_check[0]['id']


            ##################
            ## GRAB ISCROWD ## 
            ##################

            cur_iscrowd = 0

            ###################
            ## GRAB THE AREA ##
            ###################

            cur_area = 0

            ###################
            ## GRAB THE BBOX ##
            ###################

            cur_bbox = [0, 0 ,0 ,0]

            ## join all the current annotation information into a dictionary
            cur_annotation = {'id': cur_id, 'image_id':cur_image_id, 'segmentation':[cur_segmentation], 'category_id':cur_category_id, 'category_name':cur_category_name, 'iscrowd':cur_iscrowd, 'area':cur_area, 'bbox':cur_bbox}

            ## add all the current annotations to the coco annoation list
            coco['annotations'].append(cur_annotation)

            i=i+1


        j=j+1

    ##
    ## remove the initial category and add it to the coco format
    coco['categories'] = all_categories[1:]


    return(coco)


