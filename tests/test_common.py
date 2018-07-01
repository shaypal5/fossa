"""Test common fossa functionalities."""

import pytest
import pandas as pd

from fossa import LatestWindowAnomalyDetector


def test_bad_data():
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001)
    # bad df: one-level index
    with pytest.raises(ValueError):
        df = pd.DataFrame([[1], [4]], columns=['value'])
        clf.fit(df)
    # bad df: three-level index (too deep)
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            [[1], [4]], columns=['value'],
            index=pd.MultiIndex.from_tuples([[1, 2, 3], [1, 2, 4]]))
        clf.fit(df)
    # bad df: two-level index (good) by more than one column
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            [[1, 6], [4, 5]], columns=['value', 'foo'],
            index=pd.MultiIndex.from_tuples([[1, 2], [1, 3]]))
        clf.fit(df)
    # bad df: two-level index (good) w/ one column (good) but non-numeric
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            [['foo'], ['bar']], columns=['value'],
            index=pd.MultiIndex.from_tuples([[1, 2], [1, 3]]))
        clf.fit(df)
