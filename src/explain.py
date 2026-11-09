"""SHAP explanations for the LightGBM model.
LightGBM modeli için SHAP açıklamaları.
"""
import numpy as np
import pandas as pd
import shap

from src.features import FEATURES

# Feature groups for a simple, business-level summary
# Basit, iş diline uygun özet için özellik grupları
GROUPS = {
    "sales history": ["lag_7", "lag_14", "lag_28", "rmean_7", "rmean_28", "dow_mean_4w"],
    "calendar": ["dayofweek", "day", "month", "dayofyear"],
    "payday": ["is_payday", "days_since_payday"],
    "promotion": ["onpromotion", "promo_share_family"],
    "holidays": ["hol_national", "hol_local", "is_bridge", "is_workday", "is_event"],
    "store / family": ["store_nbr", "family", "store_type", "cluster"],
    "oil price": ["oil_ma28"],
}


def shap_values(booster, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP from LightGBM itself (handles categorical features correctly).
    LightGBM'in kendi kesin TreeSHAP hesabı (kategorik özellikleri doğru işler).

    Values are in log1p(sales) space: +0.1 means about +10% sales.
    Değerler log1p(satış) ölçeğinde: +0.1 yaklaşık +%10 satış demek.
    """
    contrib = booster.predict(X[FEATURES], pred_contrib=True)
    # Plots need numbers: numeric categories (store_nbr, cluster) keep their value,
    # text categories (family, store_type) become integer codes
    # Grafikler sayı ister: sayısal kategoriler (store_nbr, cluster) değerini korur,
    # metin kategoriler (family, store_type) tamsayı koduna çevrilir
    def to_num(c):
        if c.dtype.name != "category":
            return c
        return pd.to_numeric(c.astype(object), errors="coerce").fillna(pd.Series(c.cat.codes, index=c.index))
    data = X[FEATURES].apply(to_num).to_numpy(float)
    return shap.Explanation(values=contrib[:, :-1], base_values=contrib[:, -1],
                            data=data, feature_names=FEATURES)


def group_importance(exp: shap.Explanation) -> pd.Series:
    # Mean |SHAP| summed per group, as a share of the total
    # Grup başına toplanan ortalama |SHAP|, toplamın payı olarak
    imp = pd.Series(np.abs(exp.values).mean(axis=0), index=exp.feature_names)
    g = pd.Series({k: imp[v].sum() for k, v in GROUPS.items()})
    return (g / g.sum()).sort_values(ascending=False)
