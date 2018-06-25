"""Test common fossa functionalities."""

# import pytest

from fossa import LatestWindowAnomalyDetector
from fossa.utils import dummy_data


def test_base():
    num_categ = 8
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001)
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=100, max_val=1000)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=100, max_val=1000)
    clf.fit(history)
    prediction = clf.predict(new_day)[0]
    assert len(prediction) == num_categ
    for x in prediction:
        assert x in [-1, 0, 1]
