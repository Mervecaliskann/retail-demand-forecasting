"""Backtest folds and error metrics.
Backtest katmanları (fold) ve hata metrikleri.
"""
import numpy as np
import pandas as pd

# Forecast horizon in days (same as the Kaggle test period)
# Tahmin ufku, gün (Kaggle test dönemiyle aynı)
HORIZON = 16
N_FOLDS = 4
LAST_DATE = pd.Timestamp("2017-08-15")


def make_folds(last_date=LAST_DATE, horizon=HORIZON, n_folds=N_FOLDS):
    """Non-overlapping 16-day test windows, the latest one ends on `last_date`.
    Örtüşmeyen 16 günlük test pencereleri, en sonuncusu `last_date` tarihinde biter.

    cutoff = last day the forecaster is allowed to see.
    cutoff = tahmin yapanın görebileceği son gün.
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
    # WAPE = total absolute error / total sales; robust to many small series
    # WAPE = toplam mutlak hata / toplam satış; çok sayıda küçük seride de sağlam
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "WAPE": np.abs(err).sum() / np.abs(y_true).sum()}
