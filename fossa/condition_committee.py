"""Defining conditions-based committee."""

from collections import deque
from collections.abc import Mapping

from .committee import CommitteeBasedAnomalyDetectorABC


class ConditionsCommitteeAnomalyDetector(CommitteeBasedAnomalyDetectorABC):
    """Detects anomalies by the votes of a condition-based committee.

    Conditions are functions who receive two datetime objects: the first
    representing of the new time window (that to detect anomalies in) and the
    second that of a historic window, and return True if the historic window
    from the second given time should be included in the committee. E.g.
    inclusion_condition(new_dt, historic_dt).

    Parameters
    ----------
    history_delta : datetime.timedelta
        The backwatd delta of historic data to keep. Committee members are only
        selected from within this window.
    conditions : list of callable or dict of float to callable
        A list of committee-inclusion conditions, or dict mapping float weights
        to such conditions.
    """

    def __init__(self, history_delta, conditions):
        self.history_delta = history_delta
        if not isinstance(conditions, Mapping):
            conditions = {1: cond for cond in conditions}
        self.conditions = conditions
        # direction is oldest_window<-older_win<-...<-newest_window
        # so we add with deque.append() and remove with deque.popleft()
        # objects are 2-tuples (date_index_item, time_window_dataframe)
        self.window_queue = deque(iterable=[], maxlen=None)
        self.latest_fit_window_dt = None

    def _is_fitted(self):
        return self.latest_fit_window_dt is not None

    def _get_committee(self, new_window, new_window_dt):
        for dt, window in self.window_queue:
            for weight, condition in self.conditions:
                if condition(new_window_dt, dt):
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
        # set the latest time window as the last we've seen
        self.latest_fit_window_dt = sorted_df.index.levels[0][-1]
        # push all time windows into the queue
        for dt in sorted_df.index.levels[0]:
            self.window_queue.append((dt, sorted_df.loc[dt]))
        # keep popping from the left (older) side of the queue while we
        # encounter windows out of delta
        out_of_delta = True
        while out_of_delta and len(self.window_queue) > 0:
            oldest_dt, oldest_window = self.window_queue.popleft()
            oldest_delta = abs(self.latest_fit_window_dt - oldest_dt)
            out_of_delta = oldest_delta > self.history_delta
        if not out_of_delta:
            self.window_queue.appendleft((oldest_dt, oldest_window))
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
