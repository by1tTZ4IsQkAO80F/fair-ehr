from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.model_selection import train_test_split 

class CalWrapper(ClassifierMixin, BaseEstimator):
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

        X_train, X_val, y_train, y_val =  train_test_split(X, y, 
                                     train_size=0.9,
                                     test_size=0.1,
                                     shuffle=True,
                                     random_state=self.random_state,
                                     stratify=y 
                                     )

        self.estimator.fit(X_train, y_train)

        self._cccv_ = CalibratedClassifierCV(base_estimator=self.estimator,
                                            cv='prefit')
        self._cccv_.fit(X_val, y_val)

        self.fitted_ = True
        # Return the classifier
        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("CalWrapper not fitted")
        return self._cccv_.predict(X)

    def predict_proba(self, X):
        if not self.fitted_:
            raise ValueError("CalWrapper not fitted")
        return self._cccv_.predict_proba(X)
