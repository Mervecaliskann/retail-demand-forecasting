"""Inventory simulation: order-up-to policy driven by forecasts.
Stok simülasyonu: tahminlerle çalışan order-up-to (hedef seviyeye tamamlama) politikası.

THIS IS A SIMULATION. There is no real inventory data; demand = recorded sales.
BU BİR SİMÜLASYONDUR. Gerçek stok verisi yok; talep = kayıtlı satışlar.

Timing (one day) / Zamanlama (bir gün):
  1. morning: orders arrive          / sabah: siparişler gelir
  2. day: sell min(stock, demand), unmet demand is lost / gün: min(stok, talep) satılır, karşılanamayan talep kaybolur
  3. evening: review and order       / akşam: gözden geçir ve sipariş ver
An order placed on the evening of day t can be sold from day t + L + 1.
t günü akşamı verilen sipariş t + L + 1 gününden itibaren satılabilir.
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
    """For every daily cutoff, forecast the next `horizon` days with each method.
    Her günlük cutoff için, her yöntemle sonraki `horizon` günü tahmin et.

    LightGBM uses the latest model trained at a retrain cutoff <= the daily cutoff.
    LightGBM, günlük cutoff'tan önceki veya aynı gündeki en son eğitilmiş modeli kullanır.
    Returns columns: cutoff, store_nbr, family, method, fc_sum (sum over the horizon).
    Döndürülen sütunlar: cutoff, store_nbr, family, method, fc_sum (ufuk boyunca toplam).
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
    """RMSE of the protection-period forecast error, per series and method.
    Koruma dönemi tahmin hatasının RMSE'si, seri ve yöntem başına.
    RMSE (not std) so a biased forecast also gets more safety stock.
    Std değil RMSE: sistematik sapması olan tahmin de daha fazla emniyet stoğu alır.
    """
    m = fc.merge(actual, on=["cutoff"] + KEYS)
    m["e2"] = (m["fc_sum"] - m["actual_sum"]) ** 2
    return np.sqrt(m.groupby(["method"] + KEYS)["e2"].mean()).rename("sigma").reset_index()


def simulate(demand: np.ndarray, fc_sum: np.ndarray, safety: np.ndarray, lead_time: int = LEAD_TIME):
    """Order-up-to simulation with lost sales, vectorised over series.
    Kayıp satışlı order-up-to simülasyonu, seriler üzerinde vektörel.

    demand: (n_series, T) daily demand / günlük talep
    fc_sum: (n_series, T) forecast of the next PROTECTION days, made on the evening
            before each sim day (column 0 = evening before day 0)
            her simülasyon gününden önceki akşam yapılan sonraki PROTECTION gün tahmini
    safety: (n_series,) safety stock / emniyet stoğu
    """
    n, T = demand.shape
    target = fc_sum + safety[:, None]  # order-up-to level / hedef seviye
    # Steady-state start: 1 day of forecast + safety on hand, the other L days already on order.
    # Inventory position = target, the same rule for every method.
    # Durağan başlangıç: elde 1 günlük tahmin + emniyet, kalan L gün zaten yolda.
    # Envanter pozisyonu = hedef, her yöntem için aynı kural.
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
    """Simulate one method; returns demand table (series x day), sold and end-of-day stock.
    Tek bir yöntemi simüle et; talep tablosu (seri x gün), satılan ve gün sonu stok döner.
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
