import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

class OHE(TransformerMixin, BaseEstimator):
    """A one hot encoder column transformer that propagates new variable names.
    """
    def __init__(self, attribute_names, attribute_levels='auto'):
        self.attribute_names = attribute_names
        self.attribute_levels = attribute_levels

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {pd.DataFrame}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        # X = check_array(X, accept_sparse=True, dtype=None)
        self.feature_names_ = X.columns

        self.n_features_ = X.shape[1]

        self.ohe_ = make_column_transformer(
            (OneHotEncoder(sparse=False, 
                           # drop='first',
                           categories=self.attribute_levels,
                           ), 
                self.attribute_names
                ),
            remainder='drop')

        self.ohe_.fit(X)
        self.ohe_att_names_ = [an.replace('.0','') for an in 
                self.ohe_.transformers_[0][1].get_feature_names(
                self.attribute_names)]
        print('self.ohe_att_names_:',self.ohe_att_names_)
        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        # X = check_array(X, accept_sparse=True, dtype=None)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')


        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        X_ohe = pd.DataFrame(self.ohe_.transform(X), 
                            columns = self.ohe_att_names_
                            )

        df_ = X.drop(columns=self.attribute_names).join(
                X_ohe.set_index(X.index))

        return df_

