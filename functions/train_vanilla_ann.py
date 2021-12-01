################################
## ---- Run Vanilla ANNN ---- ##
################################


#Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.models import load_model
#from sklearn.metrics import accuracy_score


def train_vanilla_ann(labeled_features,num_epochs=5, num_classes=None):

    
    ###############################
    ## PREPROCESSING AND SCALING ##
    ###############################

    ## remove any nan values
    ##labeled_features = labeled_features[~np.isnan(labeled_features).any(axis=1)]
    labeled_features = labeled_features[~pd.isnull(labeled_features).any(axis=1)]

    # isolate the features and the labels from the dataframe
    features_unnorm = labeled_features.drop(labels='category_id',axis=1)
    labels_raw = labeled_features['category_id']

    ## reshape labels from a column 'vector' to row 'vector'
    labels_raw = np.array(labels_raw).reshape(len(labels_raw),1)

    ## Normalize the data (mean=0 and std=1)
    sc = StandardScaler()
    features = sc.fit_transform(features_unnorm)


    ########################
    ### ONE HOT ENCODING ###
    ########################

    ## this is super annoying and...
    ## very specific to FORNET!
    
    ## if number of classes is unknown
    if num_classes == None:
        # one hot encode the raw labels
        ohe = OneHotEncoder()
        labels = ohe.fit_transform(labels_raw).toarray()
    #

    ## if number of classes is known
    if num_classes != None:

        ## check if the background is being ignored
        bkg_ignore = False
        if np.min(labels_raw) > 0:
            bkg_ignore = True
        #

        ## adjust number of classes if ignoring background
        if bkg_ignore == True:
            num_classes = num_classes - 1
        #
        
        ## initilize an array to hold the labels
        labels = np.zeros(len(labels_raw)*num_classes).reshape(len(labels_raw),num_classes)

        i=0
        while i < len(labels):

            cur_lab = int(labels_raw[i])

            ## adjust the current label if ignoring background
            if bkg_ignore == True:
                cur_lab = cur_lab - 1
            #

            labels[i,cur_lab] = 1
            i=i+1
        #
    #

    ## split the data into a training and test set
    features_train, features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.20)
    ##features_train, features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.00001)

    ##################
    ## BUILD THE NN ##
    ##################
    ## Neural network architecture

    ## tells keras to create a model sequentially
    model = Sequential()

    ## add each layer to the ANN (Dense = fully connected)
    model.add(Dense(features.shape[1], input_dim=features.shape[1], activation="relu"))



    ##model.add(Dense(features.shape[1], activation="relu"))
    ##model.add(Dense(features.shape[1], activation="relu"))



    
    model.add(Dense(labels.shape[1], activation="softmax"))

    ## specify the loss function and optimizer using compile functions
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    #####################
    ### TRAIN THE ANN ###
    #####################

    history = model.fit(features_train, labels_train,validation_data = (features_test,labels_test), epochs=num_epochs, batch_size=64)

    ## return the model, scaling factors, and training history.
    return([model, sc, history])


