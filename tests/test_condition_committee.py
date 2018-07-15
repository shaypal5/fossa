"""Testing the ConditionsCommitteeAnomalyDetector class."""

from datetime import timedelta

from fossa import ConditionsCommitteeAnomalyDetector
from fossa.utils import dummy_data


def test_history_delta():
    DELTA_DAYS = 3
    EXPECTED_DAYS = DELTA_DAYS + 1
    clf = ConditionsCommitteeAnomalyDetector(
        history_delta=timedelta(days=DELTA_DAYS),
        conditions=[],
    )

    # dummy df of 6 consecutive days
    df = dummy_data(
        num_windows=DELTA_DAYS * 2,
        num_categories=3,
        min_val=1,
        max_val=5,
        start='1/1/2011'
    )
    expected_dts = df.index.levels[0][-EXPECTED_DAYS:]
    clf.fit(df)
    assert len(clf.window_queue) == EXPECTED_DAYS
    for dt, window in clf.window_queue:
        assert dt in expected_dts

    # dummy df of 6 consecutive days, a month later
    df2 = dummy_data(
        num_windows=DELTA_DAYS * 2,
        num_categories=3,
        min_val=1,
        max_val=5,
        start='2/1/2011'
    )
    clf.partial_fit(df2)
    expected_dts = df2.index.levels[0][-EXPECTED_DAYS:]
    assert len(clf.window_queue) == EXPECTED_DAYS
    for dt, window in clf.window_queue:
        assert dt in expected_dts


def test_delta_cond():
    WEIGHT = 12
    clf = ConditionsCommitteeAnomalyDetector(
        history_delta=timedelta(days=3),
        conditions=[],
    ).match_delta(
        delta=timedelta(days=1),
        weight=WEIGHT,
    )

    # dummy df of some consecutive days
    df = dummy_data(
        num_windows=6,
        num_categories=3,
        min_val=1,
        max_val=5,
        start='1/1/2011'
    )
    clf.fit(df)
    last_window = df.loc[df.index.levels[0][-1]]

    new_data = dummy_data(
        num_windows=1,
        num_categories=3,
        min_val=1,
        max_val=5,
        start='1/7/2011'
    )
    new_dt = new_data.index.levels[0][0]
    new_window = new_data.loc[new_dt]
    committee = list(clf._get_committee(new_window, new_dt))
    assert len(committee) == 1
    comm_weight, comm_window = committee[0]
    assert comm_weight == WEIGHT
    assert comm_window.equals(last_window)
