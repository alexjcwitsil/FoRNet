import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(unscaled_features):

    ## remove any nan values
    unscaled_features = unscaled_features[~np.isnan(unscaled_features).any(axis=1)]

    # isolate the features and the labels from the dataframe
    features_unnorm = unscaled_features.drop(labels='category_id',axis=1)

    ## Normalize the data (mean=0 and std=1)
    sc = StandardScaler()
    features = sc.fit_transform(features_unnorm)

    return(features)


