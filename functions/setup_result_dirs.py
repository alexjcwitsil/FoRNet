#################################
## ---- Setup Directories ---- ##
###                           ###
## Setup the required director-##
##-ies for the ForNet results. ##
#################################
import os

def setup_result_dirs(sigma, project_path='/home/alexwitsil/python_packages/fornet'):

    #######################
    ## TOP LEVEL RESULTS ##
    #######################
    
    ## check if there is a results dir
    cur_dirs = os.listdir(project_path)

    ## make a results directory if need be
    if 'results' not in cur_dirs:
        os.mkdir(project_path + '/results')


    #######################
    ## 2ND LEVEL RESULTS ##
    #######################
    
    ## check if there is a training and testing directories
    cur_results_dirs = os.listdir(project_path + '/results')
    
    if 'training_results' not in cur_results_dirs:
        os.mkdir(project_path + '/results/training_results')

    if 'testing_results' not in cur_results_dirs:
        os.mkdir(project_path + '/results/testing_results')

    if 'nn_models' not in cur_results_dirs:
        os.mkdir(project_path + '/results/nn_models/')


    #######################
    ## 3RD LEVEL RESULTS ##
    #######################

    ## TRAINING RESULT DIRS
    train_path = project_path + '/results/training_results/'

    ## check if there is a training blob features directory
    cur_training_dirs = os.listdir(train_path)
    
    if 'blob_features' not in cur_training_dirs:
        os.mkdir(train_path + 'blob_features/')

    if 'labeled_features' not in cur_training_dirs:
       os.mkdir(train_path + 'labeled_features/')

    if 'true_image_segmentations' not in cur_training_dirs:
       os.mkdir(train_path + 'true_image_segmentations/')

    if 'background_features' not in cur_training_dirs:
       os.mkdir(train_path + 'background_features/')


    ## TESTING RESULT DIRS
    test_path = project_path + '/results/testing_results/'

    ## check if there is a testing blob features directory
    cur_testing_dirs = os.listdir(test_path)
    
    if 'blob_features' not in cur_testing_dirs:
        os.mkdir(test_path + 'blob_features/')

    if 'labeled_features' not in cur_testing_dirs:
        os.mkdir(test_path + 'labeled_features/')

    if 'predictions' not in cur_testing_dirs:
        os.mkdir(test_path + 'predictions/')

    if 'true_image_segmentations' not in cur_testing_dirs:
        os.mkdir(test_path + 'true_image_segmentations/')


    ## NEURAL NETWORK RESULTS DIRS 
    nn_path = project_path + '/results/nn_models/'
    
    ## check if there is the current sigma directory in nn_models
    cur_nn_dirs = os.listdir(nn_path)

    if 'sig' + str(sigma) not in cur_nn_dirs:
        os.mkdir(nn_path + 'sig' + str(sigma))


    #######################
    ## 4th LEVEL RESULTS ##
    #######################

    ## TRAINING RESULTS DIRS
    
    ## blob features directory
    blob_features_path = train_path + 'blob_features/'
    cur_blob_features_dirs = os.listdir(blob_features_path)
    
    if 'sig' + str(sigma) not in cur_blob_features_dirs:
        os.mkdir(blob_features_path + 'sig' + str(sigma))

    background_features_path = train_path + 'background_features/'
    cur_background_features_dirs = os.listdir(background_features_path)
    
    if 'sig' + str(sigma) not in cur_background_features_dirs:
        os.mkdir(background_features_path + 'sig' + str(sigma))
        

    
    ## TESTING RESULTS DIRS
    
    ## blob features directory
    blob_features_path = test_path + 'blob_features/'
    cur_blob_features_dirs = os.listdir(blob_features_path)

    if 'sig' + str(sigma) not in cur_blob_features_dirs:
        os.mkdir(blob_features_path + 'sig' + str(sigma))

    
    ## PREDICTIONS RESULTS DIRS

    predictions_path = test_path + 'predictions/'
    cur_predictions_dirs = os.listdir(predictions_path)
    
    if 'sig' + str(sigma) not in cur_predictions_dirs:
        os.mkdir(predictions_path + 'sig' + str(sigma))

    if 'stacked' not in cur_predictions_dirs:
        os.mkdir(predictions_path + 'stacked')


    #######################
    ## 5th LEVEL RESULTS ##
    #######################

    ## stacked labels and probabilities
    stack_path = predictions_path + 'stacked/'
    cur_stack_dirs = os.listdir(stack_path)

    if 'labeled_images' not in cur_stack_dirs:
        os.mkdir(stack_path + 'labeled_images')

    if 'label_probabilities' not in cur_stack_dirs:
        os.mkdir(stack_path + 'label_probabilities')



