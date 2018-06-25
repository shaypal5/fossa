"""Tests for the LatestWindowAnomalyDetector."""

import pytest
from sklearn.exceptions import NotFittedError

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
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    num_new_days = 30
    many_days = dummy_data(
        num_days=num_new_days, num_categories=num_categ, min_val=100,
        max_val=1000)
    predictions = clf.predict(many_days)
    assert len(predictions) == num_categ * num_new_days
    for x in predictions.values:
        assert x in [-1, 0, 1]


def test_normalize():
    num_categ = 8
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001, normalize=True)
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=100, max_val=1000)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=100, max_val=1000)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    num_new_days = 30
    many_days = dummy_data(
        num_days=num_new_days, num_categories=num_categ, min_val=100,
        max_val=1000)
    predictions = clf.predict(many_days)
    assert len(predictions) == num_categ * num_new_days
    for x in predictions.values:
        assert x in [-1, 0, 1]


def test_diff_categ():
    num_categ_1 = 8
    num_categ_2 = 7
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001)
    history = dummy_data(
        num_days=10, num_categories=num_categ_1, min_val=100, max_val=1000)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ_2, min_val=100, max_val=1000)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == max(num_categ_1, num_categ_2)
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_errors():
    # bad p thresholds
    with pytest.raises(ValueError):
        LatestWindowAnomalyDetector(p_threshold=2)
    # bad p thresholds
    with pytest.raises(ValueError):
        LatestWindowAnomalyDetector(p_threshold=-1)
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001)
    new_day = dummy_data(
        num_days=1, num_categories=8, min_val=100, max_val=1000)
    with pytest.raises(NotFittedError):
        clf.predict(new_day)


def test_partial_fit():
    num_categ = 8
    clf = LatestWindowAnomalyDetector(p_threshold=0.00001)
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=100, max_val=1000)
    recent_history = dummy_data(
        num_days=6, num_categories=num_categ, min_val=100, max_val=1000)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=100, max_val=1000)
    clf.fit(history)
    clf.partial_fit(recent_history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]
