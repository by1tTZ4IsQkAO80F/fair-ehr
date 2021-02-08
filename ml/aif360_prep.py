from aif360.sklearn.datasets.utils import standardize_dataset
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pdb
from protected_groups import group_levels, protected_attribute_names
from .OHE import OHE

def prep_df(self, X, y=[], privileged=[], feature_names=[]):
    """Preps a dataframe for AIF360. The protected attributes are set as
    a multiindex and within the dataset, their values are one-hot-encoded for
    categorical variables.
    Parameters
    ----------
    self: the sklearn-style estimator. used to store OHE. 
    X: data (n_samples x n_features)
    y: label

    privileged: binarize the grouping according to privileged or not privileged

    """
    training = len(y) > 0

    print('entering prep_df. X.columns:',X.columns)
    print('len(y):',len(y))

    df_ = X

    # training
    if training: 
        self.init_feature_names_ = X.columns
        df_['label'] = y
    #dummy label for prediction tasks
    else: 
        df_['label'] = np.ones(len(X))

    # binarize the privileged group that has been specified 
    if len(privileged) > 0:         
        mask = np.zeros(len(df_))
        for p in privileged:
            pmask = np.ones(len(df_))
            for k,v in p.items():
                pmask = pmask & (df_[k] == v)
            mask = pmask | mask

        df_['privileged'] = mask

        df_aif360 = standardize_dataset(df_, 
                                        ['privileged'],
                                        'label',
                                        dropcols = ['privileged'])
    else: # otherwise include all intersections of groups
        # if len(y) == 0:
        #     prot_atts = 
        df_aif360 = standardize_dataset(df_, 
                                        ['SEX','RACE','ETHNICITY'],
                                        'label')

    # train a one hot encoder for race and sex
    if training:
        self.ohe_ = OHE(attribute_names = ['RACE','ETHNICITY'],
                        attribute_levels = [group_levels['RACE'],
                                            group_levels['ETHNICITY']],
                       )

        X_ = pd.DataFrame(self.ohe_.fit_transform(df_aif360.X), 
                          index = df_aif360.X.index)
        self.ohe_feature_names_ = X_.columns
    # for prediction, transform the dataset categorical features
    else:
        X_ = pd.DataFrame(self.ohe_.transform(df_aif360.X), 
                          index = df_aif360.X.index)

    y_ = df_aif360.y

    return (X_,) if len(y) == 0 else (X_, y_)
