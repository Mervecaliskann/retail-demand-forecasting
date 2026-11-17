"""Backtest folds and error metrics.
Backtest fold'ları ve hata metrikleri.
"""
import numpy as np
import pandas as pd

# Forecast horizon in days (same as the Kaggle test period)
# Tahmin ufku, gün (Kaggle test dönemiyle aynı)
HORIZON = 16
N_FOLDS = 4
LAST_DATE = pd.Timestamp("2017-08-15")


def make_folds(last_date=LAST_DATE, horizon=HORIZON, n_folds=N_FOLDS):
    """Non-overlapping 16-day test windows ending on `last_date`; cutoff = last visible day.
    `last_date` tarihinde biten örtüşmeyen 16 günlük pencereler; cutoff = görülebilen son gün.
    """
    folds = []
    for k in range(n_folds, 0, -1):
        test_end = last_date - pd.Timedelta(days=horizon * (k - 1))
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        folds.append({"fold": n_folds - k + 1,
                      "cutoff": test_start - pd.Timedelta(days=1),
                      "test_start": test_start, "test_end": test_end})
    return pd.DataFrame(folds)


def metrics(y_true, y_pred) -> dict:
    # WAPE = total absolute error / total sales
    # WAPE = toplam mutlak hata / toplam satış
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "WAPE": np.abs(err).sum() / np.abs(y_true).sum()}
