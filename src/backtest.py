"""Rolling-window backtest: retrain at every cutoff, forecast 16 days, score.
Kayan pencereli backtest: her cutoff'ta yeniden eğit, 16 gün tahmin et, puanla.
"""
import time

import pandas as pd

from src import baselines, evaluate, model

KEYS = ["date", "store_nbr", "family"]


def run_backtest(df: pd.DataFrame, feat: pd.DataFrame, folds: pd.DataFrame,
                 horizon: int = evaluate.HORIZON, verbose: bool = True) -> pd.DataFrame:
    """Return one row per (fold, model, date, store, family) with actual and forecast.
    Her (fold, model, tarih, mağaza, aile) için gerçek değer ve tahmini içeren tablo döndürür.
    """
    out = []
    for f in folds.itertuples():
        t0 = time.time()
        actual = df[(df["date"] >= f.test_start) & (df["date"] <= f.test_end)][KEYS + ["sales"]]
        hist = df[df["date"] <= f.cutoff]  # baselines only see the past / baseline'lar sadece geçmişi görür

        preds = {
            "naive": baselines.naive(hist, f.cutoff, horizon),
            "seasonal_naive": baselines.seasonal_naive(hist, f.cutoff, horizon),
        }
        # Fresh model per fold: trained only on data up to this cutoff
        # Her fold için yeni model: sadece bu cutoff'a kadarki veriyle eğitilir
        booster = model.train(feat, f.cutoff)
        preds["lightgbm"] = model.forecast_recursive(booster, feat, f.cutoff, horizon)

        for name, p in preds.items():
            m = actual.merge(p, on=KEYS, how="left").assign(fold=f.fold, model=name)
            m["h"] = (m["date"] - f.cutoff).dt.days  # forecast step 1..16 / tahmin adımı 1..16
            out.append(m)
        if verbose:
            print(f"fold {f.fold} done in {time.time() - t0:.1f}s")
    res = pd.concat(out, ignore_index=True)
    assert res["pred"].notna().all(), "missing forecasts / eksik tahmin var"
    return res


def score(res: pd.DataFrame, by=("fold", "model")) -> pd.DataFrame:
    # Metrics per group (e.g. per fold and model)
    # Grup başına metrikler (örn. fold ve model başına)
    return (res.groupby(list(by), observed=True)
               .apply(lambda g: pd.Series(evaluate.metrics(g["sales"], g["pred"])), include_groups=False)
               .reset_index())
