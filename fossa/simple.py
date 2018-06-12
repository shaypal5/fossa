"""scikit-learn classifier wrapper for fasttext."""

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class LatestWindowAnomalyDetector(BaseEstimator, ClassifierMixin):
    """Detects distribution anomalies by comparing to the latest window.

    Parameters
    ----------
    """
    def __init__(self):
        self.last_window = None

    @staticmethod
    def _validate_x(X):
        try:
            if len(X.index.levels) != 2:
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame "
                    "with a two-leveled multi-index, where the first indexes "
                    "time windows and the second level indexes class/topic "
                    "frequency per-window."
                )
            if len(X.columns) != 1:
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame with "
                    "a single column.")
            col_lbl = X.columns[0]
            if not is_numeric_dtype(X[col_lbl]):
                raise ValueError(
                    "PreviousWindowAnomalyPredictor requires a DataFrame with "
                    "a single column of a numeric dtype.")
            return X
        except AttributeError:
            raise ValueError("PreviousWindowAnomalyPredictor requires "
                             "pandas.DataFrame objects as input.")

    def fit(self, X, y=None):
        """Fits the classifier.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and a single column of a numeric dtype, giving said
            frequency.
        y : object, optional
            This parameter is ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        self._validate_x(X)
        sorted_df = X.sort_index(level=0, ascending=False)
        last_window_ix = sorted_df.index[0][0]
        last_window = sorted_df.loc[last_window_ix]
        self.last_window = last_window
        return self

    def partial_fit(self, X, y=None):
        """Fits the classifier.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and a single column of a numeric dtype, giving said
            frequency.
        y : object, optional
            This parameter is ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        return self.fit(self, X=X, y=None)

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
        if self.last_window is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))
        return np.array([
            self._clean_label(res[0][0])
            for res in self._predict(X)
        ], dtype=np.float_)
