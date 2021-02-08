# reweighing metaclassfier. 
import numpy as np
import pandas as pd
from aif360.sklearn.preprocessing import ReweighingMeta
from .aif360_prep import prep_df

class RWM(ReweighingMeta):

    def fit(self, X, y):
        self.feature_names_ = X.columns
        self.classes_ = np.unique(y)

        return super().fit(*prep_df(self, X, y, 
            # privileged = [{'SEX':1, 'RACE':6, 'ETHNICITY':1}]
            ))

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        return super().predict(*prep_df(self, X, 
            # privileged = [{'SEX':1, 'RACE':6, 'ETHNICITY':1}]
            ))

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        return super().predict_proba(*prep_df(self, X,
            # privileged = [{'SEX':1, 'RACE':6, 'ETHNICITY':1}]
            ))
