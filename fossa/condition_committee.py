"""Defining conditions-based committee."""

import inspect
from collections import deque

from .committee import CommitteeBasedAnomalyDetectorABC


def __get_add_condition_attr_doc__(condition_generator):
    doc = condition_generator.__doc__
    without_creates = doc[len('Creates'):]
    without_creates_n_returns = without_creates[0:without_creates.find(
        'Returns\n-------\n')]
    result = 'Adds' + without_creates_n_returns
    result += (
        'self : ConditionsCommitteeAnomalyDetector\n    '
        'This anomaly detector.'
    )
    return result


def __add_condition_builder_atrribute__(condition_generator):
    def _add_condition_func(self, *args, **kwargs):
        weight, cond = condition_generator(*args, **kwargs)
        # self is always a ConditionsCommitteeAnomalyDetector
        return self.add_condition(cond, weight)
    _add_condition_func.__doc__ = __get_add_condition_attr_doc__(
        condition_generator)
    _add_condition_func.__name__ = condition_generator.__name__
    _add_condition_func.__signature__ = inspect.signature(condition_generator)
    setattr(
        ConditionsCommitteeAnomalyDetector,  # obj
        condition_generator.__name__,  # name
        _add_condition_func,  # value
    )


def __load_condition_builder_attributes__(conditions_module_obj):
    for name, obj in inspect.getmembers(conditions_module_obj):
        if callable(obj):
            __add_condition_builder_atrribute__(obj)


class ConditionsCommitteeAnomalyDetector(CommitteeBasedAnomalyDetectorABC):
    """Detects anomalies by the votes of a condition-based committee.

    Conditions are functions who receive two datetime objects: the first
    representing of the new time window (that to detect anomalies in) and the
    second that of a historic window, and return True if the historic window
    from the second given time should be included in the committee. E.g.
    inclusion_condition(new_dt, historic_dt).

    Historic windows for which more than one condition apply vote multiple
    times: once for each condition they answer, with their vote weighted by the
    associated weight.

    Parameters
    ----------
    history_delta : datetime.timedelta
        The backwatd delta of historic data to keep. Committee members are only
        selected from within this window.
    conditions : list of callable or dict of float to callable
        A list of committee-inclusion conditions, or a list of pairs of weight
        and condition.
    """

    def __init__(self, history_delta, conditions):
        self.history_delta = history_delta
        try:
            if callable(conditions[0]):
                conditions = [(1, cond) for cond in conditions]
        except IndexError:
            pass
        self.conditions = conditions
        # direction is oldest_window<-older_win<-...<-newest_window
        # so we add with deque.append() and remove with deque.popleft()
        # objects are 2-tuples (date_index_item, time_window_dataframe)
        self.window_queue = deque(iterable=[], maxlen=None)
        self.latest_fit_window_dt = None

    def add_condition(self, condition, weight=None):
        """Adds the given condition to the list of committee conditions.

        Parameters
        ----------
        condition : callable
            A committee inclusion condition function (see class documentation).
        weight : int or float, optional
            The weight to assign to votes of windows answering the given
            condition. Defaults to 1.

        Returns
        -------
        self : ConditionsCommitteeAnomalyDetector
            This anomaly detector.
        """
        if weight is None:
            weight = 1
        self.conditions.append((weight, condition))
        return self

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
        self : ConditionsCommitteeAnomalyDetector
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
        self : ConditionsCommitteeAnomalyDetector
            Returns self.
        """
        return self.fit(X=X, y=None)
