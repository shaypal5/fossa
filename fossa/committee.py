"""scikit-learn classifier wrapper for fasttext."""

from abc import ABC

import pandas as pd
from strct.dicts import increment_nested_val
from sklearn.exceptions import NotFittedError

from .base import PowerDivergenceAnomalyDetectorABC
from .utils import (
    pad_windows,
    one_vs_all_dists,
)


class CommitteeBasedAnomalyDetector(PowerDivergenceAnomalyDetectorABC, ABC):
    """Detects distribution anomalies by comparing to committees of windows .

    Parameters
    ----------
    p_threshold : float
        A threshold for p values under which difference in an observed
        distribution is determined to be a significant anomaly. Has to be a
        value between 0 and 1 (inclusive).
    """

    """ ====     Inner Documentation    ====

        This class initializes the attribute `committee` to None.
        It is expected of sub-classes to implement fit, and possibly
        partial_fit, such that it builds its committee of reference time
        windows, and provide in the `self.committee` attribute an iterable
        over (weight, window) 2-tuples of committee members.

        ==== END-OF Inner Documentation ====
    """

    __doc__ += PowerDivergenceAnomalyDetectorABC._param_subdoc

    def __init__(self, p_threshold, power=None, ddof=None):
        super().__init__(
            power=power,
            ddof=ddof,
        )
        if p_threshold < 0 or p_threshold > 1:
            raise ValueError("p_threshold must be in [0,1].")
        self.p_threshold = p_threshold
        self.committee = None

    def _detect_trend(self, obs, exp):
        """Detects trends in the given distribution, using a power divergence
        test.

        Parameters
        ----------
        obs : array-like
            The observed distribution.
        exp : array-like
            The expected distribution.

        Returns
        -------
        p_value : float
            The p-value of the performed test.
        direction : int
            The direction of the trend; 1 for an upward trend, 0 for no trend
            and -1 for a downward trend.
        """
        direction = 1
        if obs[0] < exp[0]:
            direction = -1
        res = self._power_divergence_test(f_obs=obs, f_exp=exp)
        if res[1] < self.p_threshold:
            return res[1], direction
        return res[1], 0

    def _predict_trends_for_new_window(self, new_window):
        votes_by_category = {}
        for weight, committee_win in self.committee:
            new_padded, member_padded = pad_windows(new_window, committee_win)
            new_1vall = one_vs_all_dists(new_padded)
            member_1vall = one_vs_all_dists(member_padded)
            for categ in new_1vall:
                p_val, trend = self._detect_trend(
                    obs=new_1vall[categ],
                    exp=member_1vall[categ],
                )
                increment_nested_val(
                    dict_obj=votes_by_category,
                    key_tuple


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
        y : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and two columns giving trend predictions: the first
            column gives the p value for the predicted trend, while the second
            column gives the direction of the predict trend: -1 for a downward
            trend, 0 for no trend and 1 for an upward trend. The first index
            level is identical to the input dataframe, while the second level
            contains all the categories in the union of the last time window
            and the given time windows.
        """
        if self.last_window is None:
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))
        windows_to_predict = [df.loc[ix] for ix in df.index.levels[0]]
        padded_windows = pad_windows(self.last_window, *windows_to_predict)
        padded_last_win = padded_windows.pop(0)
        pred_windows = self._predict_helper(
            new_windows=padded_windows, last_window=padded_last_win)
        res_df = pd.concat(pred_windows, keys=df.index.levels[0],
                           names=df.index.names)
        res_df.columns = ['p', 'direction']
        return res_df

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
        y : pandas.DataFrame
            A pandas DataFrame with a two-leveled multi-index, the first
            indexing time and the second indexing class/topic frequency
            per-window, and a single column giving trend predictions: -1 for
            a downward trend, 0 for no trend and 1 for an upward trend. The
            first index level is identical to the input dataframe, while the
            second level contains all the categories in the union of the last
            time window and the given time windows.
        """
        res_df = self.detect_trends(df=X)
        return res_df[['direction']]
