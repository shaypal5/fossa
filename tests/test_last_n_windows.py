"""Tests for the LastNWindowsAnomalyDetector."""

import pytest
from sklearn.exceptions import NotFittedError

from fossa import LastNWindowsAnomalyDetector
from fossa.weights import (
    exp_comp_weighter,
    exp_weighter,
    first_n_uniform_weighter,
    uniform_weighter,
)
from fossa.utils import dummy_data


def test_base():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    num_new_days = 30
    many_days = dummy_data(
        num_days=num_new_days, num_categories=num_categ, min_val=1000,
        max_val=1200)
    predictions = clf.predict(many_days)
    assert len(predictions) == num_categ * num_new_days
    for x in predictions.values:
        assert x in [-1, 0, 1]


def test_diff_categ():
    num_categ_1 = 8
    num_categ_2 = 7
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ_1, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ_2, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == max(num_categ_1, num_categ_2)
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_errors():
    # bad p thresholds
    with pytest.raises(ValueError):
        LastNWindowsAnomalyDetector(
            n_windows=4,
            weights=uniform_weighter(),
            p_threshold=2,
        )
    # bad p thresholds
    with pytest.raises(ValueError):
        LastNWindowsAnomalyDetector(
            n_windows=4,
            weights=uniform_weighter(),
            p_threshold=-1,
        )
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
    )
    new_day = dummy_data(
        num_days=1, num_categories=8, min_val=1000, max_val=1200)
    with pytest.raises(NotFittedError):
        clf.predict(new_day)


def test_partial_fit():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    recent_history = dummy_data(
        num_days=6, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    clf.partial_fit(recent_history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_p_weight():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
        p_weight=True,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_non_def_power():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
        power=0,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_non_def_ddof():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=uniform_weighter(),
        p_threshold=0.001,
        power=-2,
        ddof=4,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_first_n_uniform_weighter():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=6,
        weights=first_n_uniform_weighter(4),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_exp_weighter():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=exp_weighter(3/4),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=exp_weighter(0.1),
        p_threshold=0.001,
    )
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    clf = LastNWindowsAnomalyDetector(
        n_windows=4,
        weights=exp_weighter(2.3),
        p_threshold=0.001,
    )
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def test_exp_comp_weighter():
    num_categ = 8
    n_windows = 4
    clf = LastNWindowsAnomalyDetector(
        n_windows=n_windows,
        weights=exp_comp_weighter(n=n_windows, concave_factor=1),
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    clf = LastNWindowsAnomalyDetector(
        n_windows=n_windows,
        weights=exp_comp_weighter(n=n_windows, concave_factor=3),
        p_threshold=0.001,
    )
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    clf = LastNWindowsAnomalyDetector(
        n_windows=n_windows,
        weights=exp_comp_weighter(n=n_windows, concave_factor=30),
        p_threshold=0.001,
    )
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]

    clf = LastNWindowsAnomalyDetector(
        n_windows=n_windows,
        weights=exp_comp_weighter(n=n_windows, concave_factor=None),
        p_threshold=0.001,
    )
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]


def _weight_by_inverse_days_delta(predict_window_dt, committee_window_dt):
    delta = predict_window_dt - committee_window_dt
    return 1 / abs(delta.days)


def test_weights_function():
    num_categ = 8
    clf = LastNWindowsAnomalyDetector(
        n_windows=6,
        weights=_weight_by_inverse_days_delta,
        p_threshold=0.001,
    )
    history = dummy_data(
        num_days=10, num_categories=num_categ, min_val=1000, max_val=1200)
    new_day = dummy_data(
        num_days=1, num_categories=num_categ, min_val=1000, max_val=1200)
    clf.fit(history)
    prediction = clf.predict(new_day)
    assert len(prediction) == num_categ
    for x in prediction.values:
        assert x in [-1, 0, 1]
