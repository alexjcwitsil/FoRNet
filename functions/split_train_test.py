################################################
## ---- Split Training and Testing Data ----- ##
################################################

import random
import os
from shutil import copyfile


def split_train_test(path, test_size=0.2):

    ## list the directories in the data path
    data_dirs = os.listdir(path)

    ## check if there is a testing directory
    if 'testing' not in data_dirs:
        os.mkdir(path + '/testing')
        os.mkdir(path + '/testing/raw_images')

    ## check if there is a training directory
    if 'training' not in data_dirs:
       os.mkdir(path + '/training')
       os.mkdir(path + '/training/raw_images')

    ## check if there is anything in the testing directory
    testing_images = os.listdir(path + '/testing/raw_images')
    training_images = os.listdir(path + '/training/raw_images')

    ## if there is data in either, no need to do anthing
    if len(testing_images)!= 0 or len(training_images) != 0:
        print('Testing Data Exists')
        return

    ## If no testing data exists, split the raw imagery
    print('splitting data')

    ## grab all the ONLY image files
    all_images = os.listdir(path + '/all_images')
    all_images = [f for f in all_images if '.' in f]

    ## how many total images
    total_images = len(all_images)

    ## how many training and testing images to sample
    training_total = round(total_images * (1 - test_size))
    testing_total = total_images - training_total

    
    ## shuffle the image list
    shuffled_images = all_images[:]
    random.shuffle(shuffled_images)

    ## grab the 1:training_number images from the shuffled list
    training_images = shuffled_images[0:training_total]

    ## the remaining are the testing images
    testing_images = shuffled_images[-testing_total:]

    ## define the training and testing data dirs
    testing_dir = path + '/testing/raw_images/'
    training_dir = path + '/training/raw_images/'

    ## loop over all the training images to copy the files to the training dir
    i=0
    while i < len(training_images):

        cur_train_file = training_images[i]

        ## define the source directories
        src_dir = path + '/all_images/' + cur_train_file
        dst_dir = training_dir + cur_train_file
        
        ## copy the file to the trainiing directory
        copyfile(src_dir, dst_dir)

        i=i+1
    
    ## loop over all the testing images to copy the files to the training dir
    i=0
    while i < len(testing_images):

        cur_test_file = testing_images[i]

        ## define the source directories
        src_dir = path + '/all_images/' + cur_test_file
        dst_dir = testing_dir + cur_test_file
        
        ## copy the file to the trainiing directory
        copyfile(src_dir, dst_dir)

        i=i+1
    

    return

    











