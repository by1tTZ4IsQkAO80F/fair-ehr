from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from aif360.sklearn.postprocessing import (CalibratedEqualizedOdds,
                                           PostProcessingMeta)
import numpy as np
import pandas as pd
from .aif360_prep import prep_df
from .XGB import clf as estimator
from protected_groups import single_privileged
from .RandUnder import prep


#TODO: use prefit, presplit data
class PPM(PostProcessingMeta):

    def fit(self, X, y):

        return super().fit(*prep_df(self, X, y, 
            privileged =  single_privileged 
            ))

    def predict(self, X):

        return super().predict(*prep_df(self, X, 
            privileged =  single_privileged 
            ))

    def predict_proba(self, X):

        return super().predict_proba(*prep_df(self, X,
            privileged =  single_privileged 
            ))
        

class PPMWrapper(ClassifierMixin, BaseEstimator):
    """Hacking CalibratedClassifierCV to handle feature names.
    Parameters
    ----------
    estimator: estimator  

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, estimator, random_state=0):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        X_, y_ = prep_df(self, X,y,  single_privileged )

        X_train, X_val, y_train, y_val =  train_test_split(X_, y_, 
                                     test_size=0.5,
                                     train_size=0.5,
                                     shuffle=True,
                                     random_state=self.random_state,
                                     stratify=y_
                                     )

        self.estimator.fit(X_train, y_train)

        self._ppm_ = PostProcessingMeta(estimator=self.estimator,
                         postprocessor = CalibratedEqualizedOdds(
                                            prot_attr = ['privileged']),
                         prefit=True)
        self._ppm_.estimator_ = self.estimator
        self._ppm_.fit(X_val, y_val)

        self.fitted_ = True
        # Return the classifier
        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("CalWrapper not fitted")
        # return self._ppm_.predict(*prep_df(X))
        return self._ppm_.predict(*prep_df(self, X, 
            privileged =  single_privileged 
            ))

    def predict_proba(self, X):
        if not self.fitted_:
            raise ValueError("CalWrapper not fitted")
        # return self._ppm_.predict_proba(X)
        return self._ppm_.predict_proba(*prep_df(self, X, 
            privileged =  single_privileged 
            ))

# cal_eq_odds = CalibratedEqualizedOdds(prot_attr = ['privileged'])
# CalWrapper = PPM(estimator = estimator, 
#                 postprocessor = cal_eq_odds)
# wrapper_args = {'postprocessor':cal_eq_odds}
CalWrapper = PPMWrapper
# CEO = CalibratedEqualizedOdds(prot_attr = ['privileged'])

