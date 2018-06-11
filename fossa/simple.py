"""scikit-learn classifier wrapper for fasttext."""

import os
import abc

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError


class PreviousWindowAnomalyPredictor(BaseEstimator, ClassifierMixin):
    """An abstact base class for sklearn classifier adapters for fasttext.

    Parameters
    ----------
    """
    def __init__(self):
        self.a = 0

    def __getstate__(self):
        if self.model is not None:
            model_pickle = python_fasttext_model_to_bytes(self.model)
            pickle_dict = self.__dict__.copy()
            pickle_dict['model'] = model_pickle
            return pickle_dict
        return self.__dict__

    def __setstate__(self, dicti):
        for key in dicti:
            if key == 'model':
                unpic_model = bytes_to_python_fasttext_model(dicti[key])
                setattr(self, 'model', unpic_model)
            else:
                setattr(self, key, dicti[key])

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
        return self.a

    @staticmethod
    def _validate_x(X):
        try:
            good_index_depth = len(X.index.levels) == 2
            dt_in_index = 'datetime' in X.index.names
            lbl_in_index = 'label' in X.index.names
            if not all([good_index_depth, dt_in_index, lbl_in_index]):
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame "
                    "with a two-leveled multi-index, where the datetime level "
                    "indexes time windows and the label level indexes class "
                    "/ topic frequency per-window."
                )
            if len(X.columns) != 1:
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame with "
                    "a single column.")
            col_lbl = X.columns[0]

            return X
        except AttributeError:
            raise ValueError("PreviousWindowAnomalyPredictor requires "
                             "pandas.DataFrame objects as input.")

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

    def _format_probas(self, result):
        lbl_prob_pairs = zip(result[0], result[1])
        sorted_lbl_prob_pairs = sorted(
            lbl_prob_pairs, key=lambda x: self.class_labels_.index(x[0]))
        return [x[1] for x in sorted_lbl_prob_pairs]

    def predict_proba(self, X):
        """Predict class probabilities for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        return np.array([
            self._format_probas(res)
            for res in self._predict(X, self.num_classes_)
        ], dtype=np.float_)
