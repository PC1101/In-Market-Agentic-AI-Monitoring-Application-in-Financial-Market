import pandas as pd

from windows import get_window, ALL_WINDOWS
from metrics import evaluate_window, aggregate_metrics


def test_event_detected_with_latency():
    w = get_window("quant_meltdown_2007")  # onset 2007-08-06
    alarms = [pd.Timestamp("2007-08-13")]  # 7 days after onset
    m = evaluate_window(alarms, w, tolerance_days=21)
    assert m.detected is True
    assert m.latency_days == 7
    assert m.n_true_positives == 1
    assert m.n_false_positives == 0


def test_event_missed_when_alarm_too_late():
    w = get_window("quant_meltdown_2007")
    alarms = [pd.Timestamp("2007-09-10")]  # > 21 days after onset, still in window
    m = evaluate_window(alarms, w, tolerance_days=21)
    assert m.detected is False
    assert m.latency_days is None
    assert m.n_false_positives == 1


def test_premature_alarm_is_false_positive():
    w = get_window("quant_meltdown_2007")
    alarms = [pd.Timestamp("2007-07-20")]  # before onset
    m = evaluate_window(alarms, w, tolerance_days=21)
    assert m.detected is False
    assert m.n_false_positives == 1


def test_calm_alarm_counts_as_false_positive_rate():
    w = get_window("calm_2013_2014")
    alarms = [pd.Timestamp("2013-06-03"), pd.Timestamp("2014-02-10")]
    m = evaluate_window(alarms, w, n_trading_days=500)
    assert m.n_false_positives == 2
    assert m.fpr == 2 / 500


def test_alarms_outside_window_ignored():
    w = get_window("calm_2013_2014")
    alarms = [pd.Timestamp("2020-01-01")]
    m = evaluate_window(alarms, w, n_trading_days=500)
    assert m.n_alarms == 0


def test_aggregate_precision_recall():
    quant = get_window("quant_meltdown_2007")
    calm = get_window("calm_2013_2014")
    m_event = evaluate_window([pd.Timestamp("2007-08-09")], quant, tolerance_days=21)
    m_calm = evaluate_window([pd.Timestamp("2013-06-03")], calm, n_trading_days=500)
    agg = aggregate_metrics([m_event, m_calm])
    assert agg.n_detected == 1
    assert agg.recall == 1.0
    # 1 TP, 1 FP -> precision 0.5
    assert agg.precision == 0.5
    assert agg.false_positive_rate == 1 / 500
