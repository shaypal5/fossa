"""scikit-learn classifier wrapper for fasttext."""

from abc import ABC
from collections import deque

import pandas as pd
from strct.dicts import (
    increment_nested_val,
    get_key_val_of_max,
)
from sklearn.exceptions import NotFittedError

from .base import PowerDivergenceAnomalyDetectorABC
from .utils import (
    pad_windows,
    one_vs_all_dists,
)
from .weighter import (
    uniform_weighter,
    first_n_uniform_weighter,
    exp_weighter,
    exp_comp_weighter,
)


class CommitteeBasedAnomalyDetectorABC(PowerDivergenceAnomalyDetectorABC, ABC):
    """Detects distribution anomalies by comparing to committees of windows.

    Parameters
    ----------
    p_threshold : float
        A threshold for p values under which difference in an observed
        distribution is determined to be a significant anomaly. Has to be a
        value between 0 and 1 (inclusive).
    p_weight : bool, default False
        If set to True, votes of committee members are farther weighted by the
        by the probability of their test result.
    """

    __doc__ += PowerDivergenceAnomalyDetectorABC._param_subdoc
    _param_str = 'Parameters\n    ' + ('-' * len('Parameters')) + '\n'
    _loc = __doc__.find(_param_str)
    # +4 so not to take the first four spaces
    _param_subdoc = __doc__[_loc + len(_param_str) + 4:]

    """ ====     Inner Documentation    ====

        This class calls the unimplemented method `_get_committee(new_window)`.
        It is expected of sub-classes to implement `fit`, and possibly
        `partial_fit`, such that they builds its committee of reference time
        windows, and additionaly implement `_get_committee` method such that,
        when supplied with a window to detect anomalies in, it returns an
        iterator over (weight, window) 2-tuples of committee members.

        This allows weighting schemes that are both agnostic or dependent on
        metrics and comparisons between the new window and the committee.

        ==== END-OF Inner Documentation ====
    """

    def __init__(self, p_threshold, p_weight=False, power=None, ddof=None):
        super().__init__(
            power=power,
            ddof=ddof,
        )
        if p_threshold < 0 or p_threshold > 1:
            raise ValueError("p_threshold must be in [0,1].")
        self.p_threshold = p_threshold
        self.weigh_by_p_val = p_weight

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

    def _get_committee(self, new_window):
        raise NotImplementedError(
            "This class of committee-based anomaly detection does not "
            "implement the required self._get_committee(new_window) method, "
            "and thus needs to be either extended with sub-class or ammended."
        )

    def _predict_trends_for_new_window(self, new_window):
        votes_by_category = {}
        for weight, committee_win in self._get_committe(new_window):
            new_padded, member_padded = pad_windows(new_window, committee_win)
            new_1vall = one_vs_all_dists(new_padded)
            member_1vall = one_vs_all_dists(member_padded)
            for categ in new_1vall:
                p_val, trend = self._detect_trend(
                    obs=new_1vall[categ],
                    exp=member_1vall[categ],
                )
                vote = weight
                if self.weigh_by_p_val:
                    if trend == 0:
                        vote *= p_val
                    else:
                        vote *= (1 - p_val)
                increment_nested_val(
                    dict_obj=votes_by_category,
                    key_tuple=(categ, trend),
                    value=vote,
                )
        trends = {}
        for cat in votes_by_category:
            votes = votes_by_category[cat]
            max_trend, val = get_key_val_of_max(votes)
            conf = val / sum(votes.values())
            trends[cat] = (conf, max_trend)
        trends_df = pd.DataFrame(trends, index=['conf', 'trend']).T
        return trends_df

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
        pred_windows = [
            self._predict_trends_for_new_window(window)
            for window in windows_to_predict
        ]
        res_df = pd.concat(pred_windows, keys=df.index.levels[0],
                           names=df.index.names)
        res_df.columns = ['conf', 'trend']
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


class LastNWindowsAnomalyDetector(CommitteeBasedAnomalyDetectorABC):
    """Detects distribution anomalies by comparing to the last N seen windows.

    Parameters
    ----------
    n_windows : int
        The number of last windows to include in the committee.
    weights : str, iterable or callable
        Determines the weighing scheme between
    """
    __doc__ += CommitteeBasedAnomalyDetectorABC._param_subdoc

    def __init__(self, n_windows, weights, p_threshold, p_weight=False,
                 power=None, ddof=None):
        super().__init__(
            p_threshold=p_threshold,
            p_weight=p_weight,
            power=power,
            ddof=ddof,
        )
        self.n_windows = n_windows
        # direction is newest_window->older_win->...->oldest_window
        # so we add with deque.appendleft() and remove with deque.pop()
        self.window_queue = deque(iterable=[], maxlen=self.n_windows)

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
        last_n_windows_ix = sorted_df.index[0][-self.n_windows:]
        last_n_windows = sorted_df.loc[last_n_windows_ix]
        self.last_window = last_window
        return self

    def partial_fit(self, X, y=None):
        """Incrementaly fits the classifier to the given data.

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
        return self.fit(X=X, y=None)
