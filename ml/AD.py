import numpy as np
import pandas as pd
from aif360.sklearn.inprocessing import AdversarialDebiasing
from .aif360_prep import prep_df
from protected_groups import protected_attribute_names

class AD(AdversarialDebiasing):

    def fit(self, X, y):

        return super().fit(*prep_df(self, X, y))

    def predict(self, X):
        print('AD::predict')

        return super().predict(*prep_df(self, X))

    def predict_proba(self, X):
        print('AD::predict_proba')

        return super().predict_proba(*prep_df(self, X))

    def decision_function(self, X):
        print('AD::decision_function')
        if isinstance(X, pd.DataFrame):
            return super().decision_function(X)
        else:
            if X.shape[1] == len(self.init_feature_names_):
                X = pd.DataFrame(X, columns=self.init_feature_names_)
            return super().decision_function(*prep_df(self, X))

clf = AD(prot_attr = protected_attribute_names, 
               verbose = True
               )
name = 'AD'
