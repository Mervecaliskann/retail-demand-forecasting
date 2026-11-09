"""Simple baseline forecasts. Each one only sees data up to the cutoff.
Basit baseline tahminleri. Her biri sadece cutoff tarihine kadarki veriyi görür.
"""
import numpy as np
import pandas as pd

KEYS = ["store_nbr", "family"]


def naive(df: pd.DataFrame, cutoff, horizon: int) -> pd.DataFrame:
    """Every future day = last observed value at the cutoff.
    Gelecekteki her gün = cutoff günündeki son gözlenen değer.
    """
    last = df.loc[df["date"] == cutoff, KEYS + ["sales"]]
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon)
    out = last.merge(pd.DataFrame({"date": dates}), how="cross")
    return out.rename(columns={"sales": "pred"})[["date"] + KEYS + ["pred"]]


def seasonal_naive(df: pd.DataFrame, cutoff, horizon: int, season: int = 7) -> pd.DataFrame:
    """Every future day = sales on the same weekday in the last observed week.
    Gelecekteki her gün = son gözlenen haftada aynı haftanın günündeki satış.

    For horizon > 7 we reuse that same week again (no peeking into the test period).
    Ufuk 7'den uzunsa aynı haftayı tekrar kullanırız (test dönemine bakmadan).
    """
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon)
    h = np.arange(1, horizon + 1)
    # Source date: go back whole weeks until we are at or before the cutoff
    # Kaynak tarih: cutoff'a veya öncesine düşene kadar tam haftalar geri git
    src = dates - pd.to_timedelta(season * np.ceil(h / season).astype(int), unit="D")
    mapping = pd.DataFrame({"date": dates, "src_date": src})
    hist = df.loc[df["date"].isin(src), KEYS + ["date", "sales"]].rename(
        columns={"date": "src_date", "sales": "pred"})
    out = mapping.merge(hist, on="src_date")
    return out[["date"] + KEYS + ["pred"]]
