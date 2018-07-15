"""Condition generation functions."""


def match_delta(delta, weight=None):
    """Creates a condition matching windows with the exact given delta from the
    prediction window.

    Parameters
    ----------
    delta : datetime.timedelta
        The delta to match.
    weight : int or float, optional
        The weight to assign to votes of windows answering the given
        condition. Defaults to 1.

    Returns
    -------
    int, callable
        A pair of weight and the resultig condition.
    """
    if weight is None:
        weight = 1
    def _delta_matcher(new_window_dt, committee_window_dt):
        return (new_window_dt - committee_window_dt) == delta
    return weight, _delta_matcher


def whole_delta(self, delta, weight=None):
    """Creates a condition for windows with a delta, from the prediction
    window, that is an integer multiple of the given delta.

    Parameters
    ----------
    delta : datetime.timedelta
        The delta to match.
    weight : int or float, optional
        The weight to assign to votes of windows answering the given
        condition. Defaults to 1.

    Returns
    -------
    int, callable
        A pair of weight and the resultig condition.
    """
    if weight is None:
        weight = 1
    def _whole_delta_matcher(new_window_dt, committee_window_dt):
        return ((new_window_dt - committee_window_dt) / delta).is_integer()
    return weight, _whole_delta_matcher
