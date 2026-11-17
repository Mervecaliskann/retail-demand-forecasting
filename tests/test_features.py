"""Tests for payday features. Run: python -m pytest tests  or  python tests/test_features.py
Maaş günü özellikleri için testler. Çalıştır: python -m pytest tests  veya  python tests/test_features.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import payday_features  # noqa: E402


def _since(*days):
    is_pay, since = payday_features(pd.Series(pd.to_datetime(list(days))))
    return is_pay.tolist(), since.tolist()


def test_payday_is_zero_on_15th_and_month_end():
    # 15th, 30-day month end, 31-day month end, February end, leap-year February end
    # 15'i, 30 günlük ay sonu, 31 günlük ay sonu, Şubat sonu, artık yıl Şubat sonu
    is_pay, since = _since("2017-06-15", "2017-06-30", "2017-07-31", "2017-02-28", "2016-02-29")
    assert is_pay == [1, 1, 1, 1, 1]
    assert since == [0, 0, 0, 0, 0]


def test_days_since_payday_counts_up():
    # 1st = 1 day after month end, 14th = 14, 16th = 1, 30th of a 31-day month = 15
    # 1'i = ay sonundan 1 gün sonra, 14'ü = 14, 16'sı = 1, 31 günlük ayın 30'u = 15
    is_pay, since = _since("2017-07-01", "2017-07-14", "2017-07-16", "2017-07-30")
    assert is_pay == [0, 0, 0, 0]
    assert since == [1, 14, 1, 15]


if __name__ == "__main__":
    test_payday_is_zero_on_15th_and_month_end()
    test_days_since_payday_counts_up()
    print("all tests passed")
