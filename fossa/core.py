"""scikit-learn classifier wrapper for fasttext."""

import os
import abc

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

class FossaPredictorABC(BaseEstimator, ClassifierMixin, metaclass=abc.ABCMeta):
    """An abstact base class for sklearn-compliant time-series predictors.

    Parameters
    ----------
    """
    def __init__(self):
        pass

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        # re-implementation that will preserve ft kwargs
        pass

    @staticmethod
    def _validate_x(X):
        try:
            if len(X.shape) != 2:
                raise ValueError(
                    "FastTextClassifier methods must get a two-dimensional "
                    "numpy array (or castable) as the X parameter.")
            return X
        except AttributeError:
            return FtClassifierABC._validate_x(np.array(X))

    @staticmethod
    def _validate_y(y):
        try:
            if len(y.shape) != 1:
                raise ValueError(
                    "FastTextClassifier methods must get a one-dimensional "
                    "numpy array as the y parameter.")
            return np.array(y)
        except AttributeError:
            return FtClassifierABC._validate_y(np.array(y))

    def fit(self, X, y):
        """Fits the classifier
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        self._validate_x(X)
        y = self._validate_y(y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.num_classes_ = len(self.classes_)
        self.class_labels_ = [
            '__label__{}'.format(lbl) for lbl in self.classes_]
        # Dump training set to a fasttext-compatible file
        temp_trainset_fpath = temp_dataset_fpath()
        input_col = self._input_col(X)
        dump_xy_to_fasttext_format(input_col, y, temp_trainset_fpath)
        # train
        self.model = train_supervised(
            input=temp_trainset_fpath, **self.kwargs)
        # Return the classifier
        try:
            os.remove(temp_trainset_fpath)
        except FileNotFoundError:  # pragma: no cover
            pass
        return self

    def _predict(self, X, k=1):
        # Ensure that fit had been called
        if self.model is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))

        # Input validation{
        self._validate_x(X)
        input_col = self._input_col(X)

        return [self.model.predict(text, k) for text in input_col]

    def predict(self, X):
        """Predict labels.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given inpurt samples.
        """
        return np.array([
            self._clean_label(res[0][0])
            for res in self._predict(X)
        ], dtype=np.float_)
