"""Utility methods for fossa."""

import numpy as np
import pandas as pd


def pad_windows(*windows):
    cat_union = list(set(v for df in windows for v in df.index))
    patching_df = pd.DataFrame(data=[0] * len(cat_union), index=cat_union)
    results = []
    for df in windows:
        patching_df.columns = df.columns
        results.append(df.combine_first(patching_df))
    return results


def one_vs_all_dists(df, normalize=False):
    res = {}
    for cat in df.index:
        val = float(df.loc[cat][0])
        others = float(df[df.index != cat].sum())
        arr = [val, others]
        if normalize:
            sum_arr = sum(arr)
            arr = [v/sum_arr for v in arr]
        res[cat] = arr
    return res


def dummy_data(num_days, num_categories, min_val, max_val):
    date_range = list(pd.date_range('1/1/2011', periods=num_days, freq='D'))
    date_range = date_range * num_categories
    categories = [str(v) for v in range(num_categories)]
    dummy = pd.DataFrame(index=date_range)
    dummy.sort_index(inplace=True)
    dummy['category'] = categories * num_days
    num_vals = num_days * num_categories
    dummy['value'] = np.random.randint(min_val, max_val, size=num_vals)
    dummy = dummy.reset_index().rename(columns={'index': 'date'})
    return dummy.set_index(['date', 'category'])
