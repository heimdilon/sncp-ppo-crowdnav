"""Unit tests for the training-progress ETA helpers in sncp_ppo.train.

These cover the two pure functions that back the live elapsed/ETA readout:
- _fmt_duration: seconds -> human-readable string
- _eta_seconds:  moving-average episode time * remaining episodes
"""
from sncp_ppo.train import _fmt_duration, _eta_seconds


# --- _fmt_duration -----------------------------------------------------------

def test_fmt_duration_zero():
    assert _fmt_duration(0) == "0s"


def test_fmt_duration_seconds_only():
    assert _fmt_duration(38) == "38s"


def test_fmt_duration_truncates_fractional_seconds():
    assert _fmt_duration(38.7) == "38s"


def test_fmt_duration_minutes_and_seconds():
    assert _fmt_duration(75) == "1m 15s"


def test_fmt_duration_minutes_floor():
    # 2712s = 45m 12s
    assert _fmt_duration(2712) == "45m 12s"


def test_fmt_duration_exact_hour_drops_seconds():
    # At the hour scale, seconds are noise -> dropped
    assert _fmt_duration(3600) == "1h 0m"


def test_fmt_duration_hours_and_minutes_drops_seconds():
    # 3725s = 1h 2m 5s -> "1h 2m"
    assert _fmt_duration(3725) == "1h 2m"


def test_fmt_duration_multi_hour():
    assert _fmt_duration(5 * 3600 + 12 * 60) == "5h 12m"


# --- _eta_seconds (moving-average remaining-time estimate) --------------------

def test_eta_seconds_moving_average():
    # mean([10, 20]) = 15; 15 * 5 remaining = 75
    assert _eta_seconds([10.0, 20.0], 5) == 75.0


def test_eta_seconds_constant_rate():
    assert _eta_seconds([2.0] * 10, 100) == 200.0


def test_eta_seconds_empty_history_returns_zero():
    # No samples yet -> cannot estimate; return 0 rather than crash
    assert _eta_seconds([], 50) == 0.0
