import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import linear_model
import pandas as pd
from aif360.algorithms.inprocessing import GerryFairClassifier
# from .gerryfair_classifier import GerryFairClassifier
from aif360.datasets import BinaryLabelDataset 
from .aif360_prep import prep_df
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from aif360.algorithms.inprocessing.gerryfair import clean  #.extract_df_from_ds(dataset)
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor  #.extract_df_from_ds(dataset)
import sys
import pdb
# from .Auditor import Auditor

# class GerryFairWrap(GerryFairClassifier):

#     def fit(self, X, y):

#         return super().fit(*prep_df(self, X, y ))

#     def predict(self, X):

#         return super().predict(*prep_df(self, X))

#     def predict_proba(self, X):

#         return super().predict_proba(*prep_df(self, X))

class SklearnGerryFairClassifier(ClassifierMixin, BaseEstimator):
    """A sklearn-compatible class for GerryFairClassifier.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, protected_attribute_names,
                 protected_attribute_levels='auto',
                 C=10, printflag=True, heatmapflag=False,
                 heatmap_iter=10, heatmap_path='.', max_iters=20, 
                 gamma=0.01,
                 fairness_def='FP', predictor=linear_model.LinearRegression()):
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        self.classifiers = None
        self.errors = None
        self.fairness_violations = None
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception(
                'This metric is not yet supported for learning. '
                'Metric specified: {}.'
                .format(self.fairness_def))
        self.protected_attribute_names = protected_attribute_names
        self.protected_attribute_levels = protected_attribute_levels

    def _prep(self, X, y=[]):

        if len(y)>0: # fitting
            self.ohe_ = make_column_transformer(
                (OneHotEncoder(sparse=False, 
                               drop='first',
                               categories=self.protected_attribute_levels
                               ), 
                    self.protected_attribute_names
                    ),
                remainder='drop')

            self.ohe_.fit(X)
            self.ohe_prot_att_names_ = \
                    self.ohe_.transformers_[0][1].get_feature_names(
                    self.protected_attribute_names)

        X_ohe = pd.DataFrame(self.ohe_.transform(X), 
                            columns = self.ohe_prot_att_names_
                            )

        df_ = X.drop(columns=self.protected_attribute_names).join(
                X_ohe.set_index(X.index))

        if len(y) == 0:
            # dummy labels
            df_['label'] = np.ones(len(X))
        else:
            df_['label'] = y

        dataset= BinaryLabelDataset(
                    favorable_label=1.0, 
                    unfavorable_label=0.0,
                    df=df_, 
                    label_names = ['label'],
                    protected_attribute_names = self.ohe_prot_att_names_
                    ) 

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        
        print('extracted X:',X.shape,X.columns,X[:4])
        print('label' in X.columns)
        print('extracted X_prime:',X_prime.shape,X_prime.columns,
                X_prime[:4])
        print('extracted y:',len(y),'ones:',sum([i==1 for i in y]))
        return dataset

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
        self.feature_names_ = X.columns
        GFparams = {k:v for k,v in self.get_params().items() 
                                if k not in ['protected_attribute_names',
                                             'protected_attribute_levels']
                                and '__' not in k
                                }
        self._clf_ = GerryFairClassifier(**GFparams)
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        dataset = self._prep(X,y)

        auditor = Auditor(dataset, self.fairness_def)
        n = X.shape[0]
        costs_0, costs_1, X_0 = auditor.initialize_costs(n)
        print('cost0:',len(costs_0),'total:',np.sum(costs_0))
        print('cost1:',len(costs_1),'total:',np.sum(costs_1))
        self._clf_.fit(dataset)
        self.fitted_ = True
        # Return the classifier
        return self

    def predict(self, X):
        y_pred = self._predict(X, threshold=0.5)
        return y_pred

    def predict_proba(self, X):
        y_pred_proba = self._predict(X, threshold=0)
        return y_pred_proba

    def _predict(self, X, threshold):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])
        if not self.fitted_:
            raise ValueError("SklearnGerryFairClassifier not fitted")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        dataset = self._prep(X)

        y_pred = self._clf_.predict(dataset, threshold).labels.ravel()

        if threshold == 0:
            if np.max(y_pred) > 1.0 or np.min(y_pred) < 0:
                y_pred = (y_pred - np.min(y_pred))/np.ptp(y_pred)

            y_pred = np.vstack((1-y_pred, y_pred)).T
            print('y_pred_proba:',y_pred[:10])
        # else:
        #     y_pred = np.array(y_pred, dtype=bool)
        # GerryFair predictions
        return y_pred
