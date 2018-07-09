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
        by the probability of their test result, and not only by their
        committee weights.
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

        Additionally, the

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

    def _is_fitted(self):
        raise NotImplementedError(
            "This class of committee-based anomaly detection does not "
            "implement the required self._is_fitted() method, "
            "and thus needs to be either extended with sub-class or ammended."
        )

    def _get_committee(self, new_window, new_window_dt):
        raise NotImplementedError(
            "This class of committee-based anomaly detection does not "
            "implement the required self._get_committee(new_window) method, "
            "and thus needs to be either extended with sub-class or ammended."
        )

    def _predict_trends_for_new_window(self, new_window, new_window_dt):
        """Predicts trends for the given time window.

        Parameters
        ----------
        new_window : pandas.DataFrame
            The time window for which to detect trends; a pandas DataFrame with
            a single-level index, indexing class/topic, and a single column of
            a numeric dtype, giving frequency for each class.
        new_window_dt : datetime.datetime
            The time of the given window.

        Returns
        -------
        pandas.DataFrame
            A dataframe giving the trend, and the confidence in its detection,
            for each of the classes in the given dataframe.
        """
        votes_by_category = {}
        weights_n_committee_members = self._get_committee(
            new_window, new_window_dt)
        for weight, committee_win in weights_n_committee_members:
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
            column giving the confidence for the predicted trend, the second
            giving the direction of the predict trend: -1 for a downward trend,
            0 for no trend and 1 for an upward trend. The first index level is
            identical to the input dataframe, while the second level contains
            all the categories in the union of the corresponding time window
            and all committee time windows.
        """
        if not self._is_fitted():
            raise NotFittedError("This {} instance is not fitted yet.".format(
                self.__class__.__name__))
        windows_to_predict = [(ix, df.loc[ix]) for ix in df.index.levels[0]]
        pred_windows = [
            self._predict_trends_for_new_window(window, ix)
            for ix, window in windows_to_predict
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
        return res_df[['trend']]


class LastNWindowsAnomalyDetector(CommitteeBasedAnomalyDetectorABC):
    """Detects distribution anomalies by comparing to the last N seen windows.

    Parameters
    ----------
    n_windows : int
        The number of last windows to include in the committee.
    weights : iterable over floats or callable
        Determines the weighing scheme between different windows in the
        committee. If an iterable is given, it is called n times when the
        detector is initialized to determine the set weights for committee
        members, where the first weight returned is for the most recent window,
        and the last for the least recent window. See fossa.weights for some
        weight generator functions. If a callable is given, it is called n
        times on each call to predict, giving the datetime index of the window
        to predict and of the committe member to provide a weight for; i.e.
        weights(window_to_predict_dt, committee_window_df).
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
        self.weights = weights
        # direction is oldest_window<-older_win<-...<-newest_window
        # so we add with deque.append() and remove with deque.popleft()
        # objects are 2-tuples (date_index_item, time_window_dataframe)
        self.window_queue = deque(iterable=[], maxlen=self.n_windows)

    def _is_fitted(self):
        return len(self.window_queue) > 0

    def _get_committee(self, new_window, new_window_dt):
        committee = reversed(self.window_queue)
        if callable(self.weights):
            for dt, window in committee:
                weight = self.weights(new_window_dt, dt)
                yield weight, window
        else:
            for weight, dt_n_window in zip(self.weights, committee):
                dt, window = dt_n_window
                yield weight, window

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
        sorted_df = X.sort_index(level=0, ascending=True)
        last_n_windows_ix = sorted_df.index.levels[0][-self.n_windows:]
        for ix in last_n_windows_ix:
            self.window_queue.append((ix, sorted_df.loc[ix]))
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
