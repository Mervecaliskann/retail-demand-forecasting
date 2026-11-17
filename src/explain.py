"""SHAP explanations for the LightGBM model.
LightGBM modeli için SHAP açıklamaları.
"""
import numpy as np
import pandas as pd
import shap

from src.features import FEATURES

# Feature groups for a short summary
# Kısa özet için özellik grupları
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
    """LightGBM's exact TreeSHAP, in log1p(sales) space (+0.1 is about +10% sales).
    LightGBM'in kesin TreeSHAP hesabı, log1p(satış) ölçeğinde (+0.1 yaklaşık +%10 satış).
    """
    contrib = booster.predict(X[FEATURES], pred_contrib=True)
    # Plots need numbers: numeric categories keep their value, text categories become codes
    # Grafikler sayı ister: sayısal kategoriler değerini korur, metin kategoriler koda çevrilir
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
