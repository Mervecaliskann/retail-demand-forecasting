"""Inventory simulation with an order-up-to policy. No real stock data: demand = recorded sales.
Order-up-to politikasıyla stok simülasyonu. Gerçek stok verisi yok: talep = kayıtlı satış.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import model

KEYS = ["store_nbr", "family"]
LEAD_TIME = 3
REVIEW = 1
PROTECTION = LEAD_TIME + REVIEW  # days each order must cover / her siparişin karşılaması gereken gün


def daily_forecasts(df, feat, cutoffs, retrain_cutoffs, horizon=PROTECTION, verbose=True):
    """Forecast sum of the next `horizon` days (fc_sum) per daily cutoff and method.
    Her günlük cutoff ve yöntem için sonraki `horizon` günün tahmin toplamı (fc_sum).
    """
    retrain_cutoffs = sorted(pd.to_datetime(retrain_cutoffs))
    sales = df.pivot_table(index="date", columns=KEYS, values="sales", observed=True)
    boosters, out, t0 = {}, [], time.time()
    for c in pd.to_datetime(cutoffs):
        # Naive: last value x horizon / Naive: son değer x ufuk
        naive = sales.loc[c] * horizon
        # Seasonal naive: same weekdays of the last week / Seasonal naive: son haftanın aynı günleri
        days = [c + pd.Timedelta(days=h) - pd.Timedelta(days=7 * int(np.ceil(h / 7))) for h in range(1, horizon + 1)]
        snaive = sales.loc[days].sum()
        for name, s in {"naive": naive, "seasonal_naive": snaive}.items():
            out.append(s.rename("fc_sum").reset_index().assign(cutoff=c, method=name))

        # LightGBM: retrain only at retrain cutoffs (every 16 days)
        # LightGBM: sadece yeniden eğitim cutoff'larında eğit (16 günde bir)
        rc = max(r for r in retrain_cutoffs if r <= c)
        if rc not in boosters:
            boosters = {rc: model.train(feat, rc)}  # keep one model in memory / bellekte tek model
        p = model.forecast_recursive(boosters[rc], feat, c, horizon)
        s = p.groupby(KEYS, observed=True)["pred"].sum().rename("fc_sum")
        out.append(s.reset_index().assign(cutoff=c, method="lightgbm"))
    if verbose:
        print(f"{len(cutoffs)} daily cutoffs in {time.time() - t0:.0f}s")
    res = pd.concat(out, ignore_index=True)
    res["family"] = res["family"].astype(str)
    return res


def protection_actuals(df, cutoffs, horizon=PROTECTION):
    # Real demand over the next `horizon` days after each cutoff
    # Her cutoff'tan sonraki `horizon` gündeki gerçek talep
    sales = df.pivot_table(index="date", columns=KEYS, values="sales", observed=True)
    fwd = sales[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    fwd = fwd.loc[pd.to_datetime(cutoffs)]
    fwd.index.name = "cutoff"
    a = fwd.stack(KEYS, future_stack=True).rename("actual_sum").reset_index()
    a["family"] = a["family"].astype(str)
    return a


def error_sigma(fc, actual):
    """RMSE of the 4-day forecast error per series and method (RMSE, so bias also counts).
    Seri ve yöntem başına 4 günlük tahmin hatasının RMSE'si (RMSE, sapma da sayılsın diye).
    """
    m = fc.merge(actual, on=["cutoff"] + KEYS)
    m["e2"] = (m["fc_sum"] - m["actual_sum"]) ** 2
    return np.sqrt(m.groupby(["method"] + KEYS)["e2"].mean()).rename("sigma").reset_index()


def simulate(demand: np.ndarray, fc_sum: np.ndarray, safety: np.ndarray, lead_time: int = LEAD_TIME):
    """Order-up-to with lost sales. Each day: orders arrive, sell, order in the evening (arrives day t + L + 1).
    Kayıp satışlı order-up-to. Her gün: sipariş gelir, satış, akşam sipariş (t + L + 1. gün gelir).
    """
    n, T = demand.shape
    target = fc_sum + safety[:, None]  # order-up-to level / hedef seviye
    # Start: 1 day of forecast + safety on hand, the other L days on order (same for all methods)
    # Başlangıç: elde 1 günlük tahmin + emniyet, kalan L gün yolda (tüm yöntemler için aynı)
    daily = fc_sum[:, 0] / (lead_time + 1)
    on_hand = daily + safety
    arrivals = np.zeros((n, T + lead_time + 2))
    arrivals[:, 1:lead_time + 1] = daily[:, None]
    sold, inv_end = np.zeros((n, T)), np.zeros((n, T))
    for d in range(T):
        on_hand += arrivals[:, d]
        sold[:, d] = np.minimum(on_hand, demand[:, d])
        on_hand -= sold[:, d]
        inv_end[:, d] = on_hand
        # Evening order, only if it can still arrive inside the simulation
        # Akşam siparişi, sadece simülasyon içinde gelebilecekse
        if d + 1 < T and d + lead_time + 1 < T:
            position = on_hand + arrivals[:, d + 1:].sum(axis=1)  # stock + on order / stok + yoldaki
            q = np.maximum(0, target[:, d + 1] - position)
            arrivals[:, d + lead_time + 1] += q
    return sold, inv_end


def summarize(demand, sold, inv_end) -> dict:
    # Fill rate = demand served / total demand; inventory = mean end-of-day stock
    # Fill rate = karşılanan talep / toplam talep; stok = ortalama gün sonu stoğu
    return {
        "fill_rate": sold.sum() / demand.sum(),
        "avg_inventory": inv_end.mean(axis=1).sum(),
        "days_of_cover": inv_end.mean(axis=1).sum() / demand.mean(axis=1).sum(),
        "stockout_days_%": 100 * ((demand - sold) > 1e-9).mean(),
        "lost_sales": (demand - sold).sum(),
    }


def simulate_method(df, fc_sim, sigma, sim_days, method: str, service_level: float):
    """Simulate one method; returns demand (series x day), sold and end-of-day stock.
    Tek yöntemi simüle et; talep (seri x gün), satılan ve gün sonu stok döner.
    """
    d = df[df["date"].isin(sim_days)].copy()
    d["family"] = d["family"].astype(str)
    demand = d.pivot_table(index=KEYS, columns="date", values="sales")
    g = fc_sim[fc_sim["method"] == method]
    fc = g.pivot_table(index=KEYS, columns="cutoff", values="fc_sum").reindex(demand.index)
    # Pad cutoffs after the last useful order (not used for ordering)
    # Son yararlı siparişten sonraki cutoff'ları doldur (sipariş için kullanılmaz)
    fc = fc.reindex(columns=pd.to_datetime(sim_days) - pd.Timedelta(days=1)).ffill(axis=1)
    sig = sigma[sigma["method"] == method].set_index(KEYS)["sigma"].reindex(demand.index).fillna(0)
    safety = norm.ppf(service_level) * sig.to_numpy()
    sold, inv_end = simulate(demand.to_numpy(), fc.to_numpy(), safety)
    return demand, sold, inv_end


def run(df, fc_sim, sigma, sim_days, service_level: float) -> pd.DataFrame:
    """Simulate every method at one service level.
    Tüm yöntemleri tek bir servis seviyesinde simüle et.
    """
    rows = []
    for method in sorted(fc_sim["method"].unique()):
        demand, sold, inv_end = simulate_method(df, fc_sim, sigma, sim_days, method, service_level)
        rows.append({"method": method, "service_level": service_level,
                     **summarize(demand.to_numpy(), sold, inv_end)})
    return pd.DataFrame(rows)
