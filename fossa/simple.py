"""scikit-learn classifier wrapper for fasttext."""

from scipy.stats import chisquare
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from .utils import (
    pad_windows,
    one_vs_all_dists,
)


class LatestWindowAnomalyDetector(BaseEstimator, ClassifierMixin):
    """Detects distribution anomalies by comparing to the latest window.

    Parameters
    ----------
    p_threshold : float
        A threshold for p values under which difference in an observed
        distribution is determined to be a significant anomaly. Has to be a
        value between 0 and 1 (inclusive).
    """

    def __init__(self, p_threshold):
        if p_threshold < 0 or p_threshold > 1:
            raise ValueError("p_threshold must be in [0,1].")
        self.p_threshold = p_threshold
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

    def _detect_trend(self, obs, exp):
        direction = 1
        if obs[0] < exp[0]:
            direction = -1
        res = chisquare(obs, exp)
        if res[1] < self.p_threshold:
            return res[1], direction
        return res[1], 0

    def _predict_helper(self, new_windows, last_window):
        padded = pad_windows(last_window, *new_windows)
        padded_last = padded.pop(0)
        last_1vall = one_vs_all_dists(padded_last)
        res = []
        for new_win in padded:
            new_1vall = one_vs_all_dists(new_win)
            pred = {
                cat: self._detect_trend(new_1vall[cat], last_1vall[cat])
                for cat in new_1vall
            }
            res.append(pd.DataFrame(pred, index=['p', 'direction']).T)
        return res

    def detect_trends(self, df):
        """Detect trends in input data.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and a single column of a numeric dtype, giving said
            frequency.

        Returns
        -------
        y : array of pandas.DataFrame objects
            Predicted labels f
        """
        if self.last_window is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))
        windows_to_predict = [df.loc[ix] for ix in df.index.levels[0]]
        padded_windows = pad_windows(self.last_window, *windows_to_predict)
        padded_last_win = padded_windows.pop(0)
        return self._predict_helper(
            new_windows=padded_windows, last_window=padded_last_win)

    def predict(self, X):
        """Detect trends in input data.

        Parameters
        ----------
        X : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and a single column of a numeric dtype, giving said
            frequency.

        Returns
        -------
        y : array of pandas.DataFrame objects
            Predicted labels f
        """
        res = self.detect_trends(df=X)
        return [df.iloc[:, 1] for df in res]
