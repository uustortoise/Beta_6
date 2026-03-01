import pytest

from ml.beta6.evaluation_engine import assert_no_leakage, evaluate_leakage
from ml.beta6.feature_store import has_resident_leakage, has_time_leakage, has_window_overlap


def test_has_resident_leakage_false_when_disjoint():
    assert not has_resident_leakage(["A", "B"], ["C", "D"])


def test_has_resident_leakage_true_when_overlap():
    assert has_resident_leakage(["A", "B"], ["B", "C"])


def test_has_time_leakage_true_for_same_resident_overlap():
    train = [("A", 0, 100)]
    valid = [("A", 90, 140)]
    assert has_time_leakage(train, valid)


def test_has_time_leakage_false_for_different_residents():
    train = [("A", 0, 100)]
    valid = [("B", 50, 120)]
    assert not has_time_leakage(train, valid)


def test_has_window_overlap_true_when_gap_buffer_violated():
    train = [("A", 0, 100)]
    valid = [("A", 105, 130)]
    assert has_window_overlap(train, valid, gap_seconds=10)


def test_evaluate_leakage_reports_all_flags():
    train_residents = ["A", "B"]
    valid_residents = ["B", "C"]
    train_windows = [("B", 0, 100)]
    valid_windows = [("B", 90, 110)]
    report = evaluate_leakage(
        train_resident_ids=train_residents,
        validation_resident_ids=valid_residents,
        train_windows=train_windows,
        validation_windows=valid_windows,
        gap_seconds=5,
    )
    assert report.resident_overlap
    assert report.time_overlap
    assert report.window_overlap
    assert report.has_any_leakage


def test_assert_no_leakage_raises_on_report():
    report = evaluate_leakage(
        train_resident_ids=["A"],
        validation_resident_ids=["A"],
        train_windows=[("A", 0, 100)],
        validation_windows=[("A", 50, 80)],
        gap_seconds=5,
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(report)
