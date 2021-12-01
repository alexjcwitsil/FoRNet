###################################################
## ---- Setup K-fold Validation Directories ---- ##
###################################################

import os
import fornet as fn
import numpy as np
import itertools 
import string
import shutil
import subprocess


def setup_kfold_dirs(gaus_sigs, kfold_dir, source_dir):

    ## define gaussian sigma values to use
    gaus_sig_line = 'gaus_sigs = ' + str(gaus_sigs)

    ###############
    ## FIND DATA ##
    ###############

    all_data_dir = kfold_dir + 'all_data/data/all_images/'
    all_data_files = os.listdir(all_data_dir)

    k_files_all = np.sort(list(itertools.permutations(all_data_files, 2)))
    k_files = np.unique(k_files_all, axis=0)

    i=0
    while i < len(k_files):

        # ## what are the current training and testing files
        cur_testing_files = k_files[i]

        # ## find the index locations of the current testing files in all the datafiles
        cur_training_files = [i for i in all_data_files if i not in cur_testing_files]


        # ## make the directory if it doesn't already exist
        cur_project_path = kfold_dir + 'kfold_' + str(i).zfill(2) + '/'
        if os.path.isdir(cur_project_path) == False:
            os.mkdir(cur_project_path)
        #

        ## make a data directory
        cur_data_dir = cur_project_path + 'data/'
        if os.path.isdir(cur_data_dir) == False:
            os.mkdir(cur_data_dir)
        #

        ## make testing and training directories
        cur_testing_dir = cur_data_dir + 'testing/'
        if os.path.isdir(cur_testing_dir) == False:
            os.mkdir(cur_testing_dir)
        #
        cur_training_dir = cur_data_dir + 'training/'
        if os.path.isdir(cur_training_dir) == False:
            os.mkdir(cur_training_dir)
        #

        ## make raw images directories
        cur_testing_raw_images_dir = cur_testing_dir + 'raw_images/'
        if os.path.isdir(cur_testing_raw_images_dir) == False:
            os.mkdir(cur_testing_raw_images_dir)
        #

        ## make raw images directories
        cur_training_raw_images_dir = cur_training_dir + 'raw_images/'
        if os.path.isdir(cur_training_raw_images_dir) == False:
            os.mkdir(cur_training_raw_images_dir)
        #

        ## copy over the training and testing data
        [shutil.copy(all_data_dir + i, cur_testing_raw_images_dir + i) for i in cur_testing_files]
        [shutil.copy(all_data_dir + i, cur_training_raw_images_dir + i) for i in cur_training_files]

        ## copy the label info as well 
        label_info_path = source_dir + 'data/label_info.json'
        shutil.copy(label_info_path, cur_data_dir)


        #################################
        ### OTHER THINGS YOU MAY NEED ###
        #################################

        ## copy the scripts
        script_dir = cur_project_path + 'scripts/'
        if os.path.isdir(script_dir) == False:
            os.mkdir(script_dir)
        #
        src_script_dir = source_dir + 'scripts/'
        script_files_all = [f for f in os.listdir(src_script_dir) if os.path.isfile(os.path.join(src_script_dir,f))]
        script_files = [i for i in script_files_all if '~' not in i and '#' not in i]

        ## copy the scripts 
        [shutil.copy(src_script_dir + i, script_dir + i) for i in script_files]


        ###################
        ## AMEND SCRIPTS ##
        ###################

        ## add header line to script
        header = '#!/usr/bin/env python3'

        run_fornet_filename = cur_project_path + '/scripts/run_fornet.py'

        ## read in the script lines
        with open(run_fornet_filename) as f:
            lines_string = f.read()
            lines = lines_string.split('\n')
            first = lines[0]

        #

        ## find the guas sig line
        gaus_sigs_ind = [lines.index(i) for i in lines if 'gaus_sigs = ' in i]

        ## replace the gaus sig line wit the inputed gaus_sig_line
        lines[gaus_sigs_ind[0]] = gaus_sig_line 

        ## define the project path line
        project_path_line = 'project_path = ' + '\'' + cur_project_path + '\''

        ## find the project path line
        project_path_ind = [lines.index(i) for i in lines if 'project_path = ' in i]

        ## replace the project path line with the correct project path
        lines[project_path_ind[0]] = project_path_line









        # ## temporarily comment out the training line
        # old_training_line = 'fn.train(project_path, gaus_sigs,bkg_ignore=True)'
        # new_training_line = '######## ' + old_training_line

        # train_line_ind = [lines.index(i) for i in lines if old_training_line in i]
        # lines[train_line_ind[0]] = new_training_line















        ## write these lines to the file
        textfile = open(run_fornet_filename, "w")

        for element in lines:
            textfile.write(element + "\n")
        #
        textfile.close()


        ## first check if the header already exists
        if first != header:
            with open(run_fornet_filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)
            #
        #

        ## make sure the run_fornet script is executable
        make_exec_cmd = ['chmod', '+x', cur_project_path + '/scripts/run_fornet.py']
        subprocess.run(make_exec_cmd)


        ##################
        ### UNET STUFF ###
        ##################

        ## copy the unet directory
        unet_dir = cur_project_path + 'unet/'
        if os.path.isdir(unet_dir) == False:
            os.mkdir(unet_dir)
        #

        ## the unet data dir 
        unet_data_dir = unet_dir + 'data/'
        if os.path.isdir(unet_data_dir) == False:
            os.mkdir(unet_data_dir)
        #

        ## the unet testing and training dirs
        unet_testing_dir = unet_data_dir + 'testing/'
        unet_training_dir = unet_data_dir + 'training/'

        if os.path.isdir(unet_testing_dir) == False:
            os.mkdir(unet_testing_dir)
        #
        if os.path.isdir(unet_training_dir) == False:
            os.mkdir(unet_training_dir)
        #

        ## unet training masks 
        unet_training_masks_dir = unet_training_dir + 'masks/'
        if os.path.isdir(unet_training_masks_dir) == False:
            os.mkdir(unet_training_masks_dir)
        #

        ## unet training raw 
        unet_training_raw_dir = unet_training_dir + 'raw/'
        if os.path.isdir(unet_training_raw_dir) == False:
            os.mkdir(unet_training_raw_dir)
        #

        ## unet testing masks 
        unet_testing_masks_dir = unet_testing_dir + 'masks/'
        if os.path.isdir(unet_testing_masks_dir) == False:
            os.mkdir(unet_testing_masks_dir)
        #

        ## unet testing raw 
        unet_testing_raw_dir = unet_testing_dir + 'raw/'
        if os.path.isdir(unet_testing_raw_dir) == False:
            os.mkdir(unet_testing_raw_dir)
        #

        ## copy the data files to the UNET directories
        [shutil.copy(all_data_dir + i, unet_testing_raw_dir + i) for i in cur_testing_files]
        [shutil.copy(all_data_dir + i, unet_training_raw_dir + i) for i in cur_training_files]

        ## copy the label info as well
        label_info_path = source_dir + '/unet/data/label_info.json'
        shutil.copy(label_info_path, unet_data_dir)


        ## unet result dir
        unet_results_dir = unet_dir + 'results/'
        if os.path.isdir(unet_results_dir) == False:
            os.mkdir(unet_results_dir)
        #

        ## unet results model dir
        unet_result_model_dir = unet_results_dir + 'models/'
        if os.path.isdir(unet_result_model_dir) == False:
            os.mkdir(unet_result_model_dir)
        #

        ## unet results testing_results dir
        unet_result_testing_results_dir = unet_results_dir + 'testing_results/'
        if os.path.isdir(unet_result_testing_results_dir) == False:
            os.mkdir(unet_result_testing_results_dir)
        #

        ## unet testing result predictions
        unet_result_testing_results_predictions_dir = unet_result_testing_results_dir + 'predictions/'
        if os.path.isdir(unet_result_testing_results_predictions_dir) == False:
            os.mkdir(unet_result_testing_results_predictions_dir)
        #    


        ###########################
        ## AMEND RUN UNET SCRIPT ##
        ###########################

        ## define the project path line
        project_path_line = 'project_dir = ' + '\'' + cur_project_path + '\''

        ## add header line to script
        header = '#!/usr/bin/env python3'

        run_unet_filename = cur_project_path + '/scripts/run_unet.py'

        ## read in the script lines
        with open(run_unet_filename) as f:
            lines_string = f.read()
            lines = lines_string.split('\n')
            first = lines[0]

        #

        ## find the project path line
        project_path_ind = [lines.index(i) for i in lines if 'project_dir = ' in i]

        ## replace the project path line with the correct project path
        lines[project_path_ind[0]] = project_path_line

        ## write these lines to the file
        textfile = open(run_unet_filename, "w")
        for element in lines:
            textfile.write(element + "\n")
        #
        textfile.close()

        ## add header to executable script
        ## first check if the header already exists
        if first != header:
            with open(run_unet_filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)
            #
        #


        ##############################
        ### AMEND TEST UNET SCRIPT ###
        ##############################

        test_unet_filename = cur_project_path + '/scripts/test_unet.py'

        ## read in the script lines
        with open(test_unet_filename) as f:
            lines_string = f.read()
            lines = lines_string.split('\n')
            first = lines[0]
        #

        ## define the project path line
        project_path_line = 'project_path = ' + '\'' + cur_project_path + '\''

        ## find the project path line
        project_path_ind = [lines.index(i) for i in lines if 'project_path = ' in i]

        ## replace the project path line with the correct project path
        lines[project_path_ind[0]] = project_path_line

        ## write these lines to the file
        textfile = open(test_unet_filename, "w")
        for element in lines:
            textfile.write(element + "\n")
        #
        textfile.close()

        ## add header to executable script
        ## first check if the header already exists
        if first != header:
            with open(test_unet_filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)
            #
        #


        ################################
        ## AMEND TRUE MASKS UNET FILE ##
        ################################

        gen_true_masks_filename = cur_project_path + '/scripts/gen_true_masks.py'

            ## read in the script lines
        with open(gen_true_masks_filename) as f:
            lines_string = f.read()
            lines = lines_string.split('\n')
            first = lines[0]

        #

        ## define the project path line
        project_path_line = 'project_path = ' + '\'' + cur_project_path + 'unet/' +  '\''

        ## find the project path line
        project_path_ind = [lines.index(i) for i in lines if 'project_path = ' in i]

        ## replace the project path line with the correct project path
        lines[project_path_ind[0]] = project_path_line

        ## write these lines to the file
        textfile = open(gen_true_masks_filename, "w")
        for element in lines:
            textfile.write(element + "\n")
        #
        textfile.close()

        ## add header to executable script
        ## first check if the header already exists
        if first != header:
            with open(gen_true_masks_filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(header.rstrip('\r\n') + '\n' + content)
            #
        #



        ## make sure the run_unet scripts are executable
        make_exec_cmd1 = ['chmod', '+x', cur_project_path + '/scripts/run_unet.py']
        make_exec_cmd2 = ['chmod', '+x', cur_project_path + '/scripts/gen_true_masks.py']
        make_exec_cmd3 = ['chmod', '+x', cur_project_path + '/scripts/test_unet.py']
        subprocess.run(make_exec_cmd1)
        subprocess.run(make_exec_cmd2)
        subprocess.run(make_exec_cmd3)


        i=i+1
    #
#








