"""LightGBM training and recursive 16-day forecasting.
LightGBM eğitimi ve recursive (adım adım) 16 günlük tahmin.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURES, CAT_FEATURES, KEYS, add_lag_features

# Fixed, modest settings: no tuning on the test folds
# Sabit, ölçülü ayarlar: test fold'ları üzerinde ayar yapılmadı
PARAMS = {
    "objective": "regression",   # L2 on log1p(sales) / log1p(satış) üzerinde L2
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}
NUM_ROUNDS = 600

# Enough history for the longest feature (lag 28 + rolling 28)
# En uzun özellik için yeterli geçmiş (lag 28 + rolling 28)
HISTORY_DAYS = 60


def training_rows(feat: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Rows usable for training: up to the cutoff, all lags present,
    not a closed day, not in the earthquake weeks.
    Eğitimde kullanılabilecek satırlar: cutoff'a kadar, tüm lag'ler dolu,
    kapalı gün değil, deprem haftalarında değil.
    """
    m = (feat["date"] <= cutoff) & feat["rmean_28"].notna()
    m &= (feat["is_closed"] == 0) & (feat["quake_period"] == 0)
    return feat[m]


def train(feat: pd.DataFrame, cutoff) -> lgb.Booster:
    tr = training_rows(feat, cutoff)
    ds = lgb.Dataset(tr[FEATURES], label=np.log1p(tr["sales"]),
                     categorical_feature=CAT_FEATURES, free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)


def forecast_recursive(model: lgb.Booster, feat: pd.DataFrame, cutoff, horizon: int) -> pd.DataFrame:
    """Predict day 1, write it back as 'sales', rebuild lags, predict day 2, ...
    1. günü tahmin et, 'sales' olarak geri yaz, lag'leri yeniden kur, 2. günü tahmin et, ...

    Exogenous columns (calendar, promo, holidays) for future days are known in advance.
    Gelecek günlerin dış değişkenleri (takvim, promosyon, tatil) önceden bilinir.
    """
    start = cutoff - pd.Timedelta(days=HISTORY_DAYS)
    end = cutoff + pd.Timedelta(days=horizon)
    w = feat[(feat["date"] > start) & (feat["date"] <= end)].copy()
    w = w.sort_values(KEYS + ["date"]).reset_index(drop=True)
    future = w["date"] > cutoff
    w["sales"] = w["sales"].astype("float64")
    w.loc[future, "sales"] = np.nan  # hide the truth / gerçeği gizle

    for day in pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon):
        w = add_lag_features(w)
        rows = w["date"] == day
        pred = np.expm1(model.predict(w.loc[rows, FEATURES]))
        w.loc[rows, "sales"] = np.clip(pred, 0, None)

    out = w.loc[future, ["date"] + KEYS + ["sales"]].rename(columns={"sales": "pred"})
    return out.reset_index(drop=True)
