from xgboost import XGBClassifier 
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from protected_groups import single_privileged
from .aif360_prep import prep_df

class XGBdf(XGBClassifier):
    """XGBoost wrapper that enforces dataframe usage and column name saving"""

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            self.enforce_df_ = True
        else:
            self.enforce_df_ = False

        return super().fit(X, y)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame) and self.enforce_df_:
            X = pd.DataFrame(X, columns=self.feature_names_)

        return super().predict(X)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) and self.enforce_df_:
            X = pd.DataFrame(X, columns=self.feature_names_)

        return super().predict_proba(X)

# parameter variation
hyper_params = {
        'n_estimators': [500],
        'max_depth': [3, 6],
        'learning_rate':[0.01, 0.1, 0.3]
    }

cv = StratifiedKFold(n_splits=3, shuffle=False)
# create the classifier
# base_clf = XGBClassifier(n_jobs=1)
base_clf = XGBdf(n_jobs=1)

clf = GridSearchCV(base_clf, 
                   cv=cv, 
                   param_grid=hyper_params,
                   verbose=1
                   )

name = 'XGB'
