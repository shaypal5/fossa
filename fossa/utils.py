"""Utility methods for fossa."""

import numpy as np
import pandas as pd


def pad_windows(*windows):
    """Pads the class distributions of given time windows with their union.

    Receives multiple dataframes as separate arguments, not in a list.
    E.g. pad_windows(df1, df2), pad_windows(df1, df2, df3), etc.

    Parameters
    ----------
    *windows : variable number of pandas.DataFrame arguments
        The time windows to pad.

    Returns
    -------
    list of pandas.DataFrame
        The padded time windows.

    Example
    -------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame(data=[[3],[1]], columns=['val'])
    >>> df1
       val
    0    3
    1    1
    >>> df2 = pd.DataFrame(data=[[2],[5],[8]], columns=['val'])
    >>> df2
       val
    0    2
    1    5
    2    8
    >>> df1_pad, df2_pad = pad_windows(df1, df2)
    >>> df1_pad
       val
    0  3.0
    1  1.0
    2  0.0
    """
    cat_union = list(set(v for df in windows for v in df.index))
    if all([len(cat_union) == len(df) for df in windows]):
        return list(windows)
    patching_df = pd.DataFrame(data=[0] * len(cat_union), index=cat_union)
    results = []
    for df in windows:
        patching_df.columns = df.columns
        results.append(df.combine_first(patching_df))
    return results


def one_vs_all_dists(df):
    res = {}
    for cat in df.index:
        val = float(df.loc[cat][0])
        others = float(df[df.index != cat].sum())
        res[cat] = [val, others]
    return res


def dummy_data(num_windows, num_categories, min_val, max_val, start=None,
               freq=None):
    """Generates dummy data in a fossa-compliant format.

    Parameters
    ----------
    num_windows : int
        The number of time windows to generate.
    num_categories : int
        The number of categories with frequency data per time window.
    min_val : int
        The minimal frequency value for a category in a time window.
    max_val : int
        The maximal frequency value for a category in a time window.
    start : str or datetime-like, optional
        The start date or time to use. See `start` param for pandas.date_range.
    freq : str or DateOffset, default 'D' (calendar daily)
        The frequency of time windows. See `freq` param for pandas.date_range.
    """
    if start is None:
        start = '1/1/2011'
    if freq is None:
        freq = 'D'
    date_range = list(pd.date_range(
        start=start, periods=num_windows, freq=freq))
    date_range = date_range * num_categories
    categories = [str(v) for v in range(num_categories)]
    dummy = pd.DataFrame(index=date_range)
    dummy.sort_index(inplace=True)
    dummy['category'] = categories * num_windows
    num_vals = num_windows * num_categories
    dummy['value'] = np.random.randint(min_val, max_val, size=num_vals)
    dummy = dummy.reset_index().rename(columns={'index': 'date'})
    return dummy.set_index(['date', 'category'])
