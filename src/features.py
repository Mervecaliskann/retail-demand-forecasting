"""Feature engineering. Every feature only uses information known at forecast time.
Özellik mühendisliği. Her özellik sadece tahmin anında bilinen bilgiyi kullanır.
"""
import numpy as np
import pandas as pd

from src import data

KEYS = ["store_nbr", "family"]

# 2016-04-16 earthquake: sales were abnormal for about a month (relief buying)
# 16.04.2016 depremi: satışlar yaklaşık bir ay anormaldi (yardım alımları)
QUAKE_START, QUAKE_END = pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-16")

# Future oil prices are unknown at forecast time: use values at least 16 days old
# Gelecekteki petrol fiyatı tahmin anında bilinmez: en az 16 gün eski değerleri kullan
OIL_LAG = 16

LAGS = [7, 14, 28]
ROLL_WINDOWS = [7, 28]

CAT_FEATURES = ["store_nbr", "family", "store_type", "cluster"]
FEATURES = (
    CAT_FEATURES
    + [f"lag_{k}" for k in LAGS]
    + [f"rmean_{w}" for w in ROLL_WINDOWS] + ["dow_mean_4w"]
    + ["dayofweek", "day", "month", "dayofyear", "is_payday", "days_since_payday"]
    + ["onpromotion", "promo_share_family"]
    + ["hol_national", "hol_local", "is_bridge", "is_workday", "is_event"]
    + ["oil_ma28"]
)


def holiday_table(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Holiday flags per (date, store_nbr); a moved holiday counts on its Transfer date.
    (tarih, mağaza) başına tatil bayrakları; taşınan tatil Transfer tarihinde sayılır.
    """
    # Drop official dates of moved holidays and earthquake events (those weeks are excluded)
    # Taşınan tatillerin resmi tarihini ve deprem olaylarını at (o haftalar zaten çıkarılıyor)
    h = holidays[~holidays["transferred"]].copy()
    h = h[~h["description"].str.contains("Terremoto")]

    # Match each holiday to stores: national = all, regional = state, local = city
    # Her tatili mağazalarla eşle: ulusal = hepsi, bölgesel = eyalet, yerel = şehir
    s = stores[["store_nbr", "city", "state"]]
    nat = h[h["locale"] == "National"].merge(s[["store_nbr"]], how="cross")
    reg = h[h["locale"] == "Regional"].merge(s, left_on="locale_name", right_on="state")
    loc = h[h["locale"] == "Local"].merge(s, left_on="locale_name", right_on="city")
    allh = pd.concat([nat, reg, loc], ignore_index=True)
    # Days off; Bridge (extra day off) and Work Day (weekend turned workday) get own flags
    # İzin günleri; Bridge (ek izin) ve Work Day (iş gününe dönen hafta sonu) ayrı bayrak
    off = allh["type"].isin(["Holiday", "Transfer", "Additional"])

    allh["hol_national"] = (off & (allh["locale"] == "National")).astype("int8")
    allh["hol_local"] = (off & (allh["locale"] != "National")).astype("int8")
    allh["is_bridge"] = (allh["type"] == "Bridge").astype("int8")
    allh["is_workday"] = (allh["type"] == "Work Day").astype("int8")
    allh["is_event"] = (allh["type"] == "Event").astype("int8")
    flags = ["hol_national", "hol_local", "is_bridge", "is_workday", "is_event"]
    # Several rows can fall on the same day: keep the max of each flag
    # Aynı güne birden çok satır düşebilir: her bayrağın maksimumunu al
    return allh.groupby(["date", "store_nbr"], as_index=False)[flags].max()


def payday_features(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Paydays are the 15th and the last day of the month; returns (is_payday, days_since_payday).
    Maaş günleri ayın 15'i ve son günü; (is_payday, days_since_payday) döndürür.
    """
    d = dates.dt
    is_payday = ((d.day == 15) | d.is_month_end).astype("int8")
    # Day 1-14: days since last month's end; day 15+: days since the 15th; month end: 0
    # 1-14. gün: geçen ay sonundan beri; 15. gün ve sonrası: 15'ten beri; ay sonu: 0
    since = np.where(d.is_month_end, 0, np.where(d.day >= 15, d.day - 15, d.day))
    return is_payday, pd.Series(since, index=dates.index, dtype="int16")


def add_exogenous(df: pd.DataFrame) -> pd.DataFrame:
    """Features that do not depend on sales: calendar, stores, holidays, promo, oil.
    Satıştan bağımsız özellikler: takvim, mağaza, tatil, promosyon, petrol.
    """
    stores = data.load_stores().rename(columns={"type": "store_type"})
    out = df.merge(stores[["store_nbr", "store_type", "cluster"]], on="store_nbr", how="left")

    # Calendar / Takvim
    d = out["date"].dt
    out["dayofweek"], out["day"], out["month"], out["dayofyear"] = d.dayofweek, d.day, d.month, d.dayofyear

    out["is_payday"], out["days_since_payday"] = payday_features(out["date"])

    # Holidays / Tatiller
    hol = holiday_table(data.load_holidays(), data.load_stores())
    out = out.merge(hol, on=["date", "store_nbr"], how="left")
    flags = ["hol_national", "hol_local", "is_bridge", "is_workday", "is_event"]
    out[flags] = out[flags].fillna(0).astype("int8")

    # Promotions are planned ahead, so same-day values are allowed
    # Promosyonlar önceden planlanır, aynı günün değeri kullanılabilir
    out["promo_share_family"] = out.groupby(["date", "family"], observed=True)["onpromotion"].transform(
        lambda x: (x > 0).mean())

    # Oil: daily calendar + forward-fill, 28-day mean, shifted 16 calendar days (not rows)
    # Petrol: günlük takvim + ileri doldurma, 28 günlük ortalama, 16 takvim günü kaydırma (satır değil)
    oil = data.load_oil().set_index("date")["dcoilwtico"].asfreq("D").ffill()
    oil_ma = oil.rolling(28, min_periods=1).mean().shift(OIL_LAG, freq="D").rename("oil_ma28")
    out = out.merge(oil_ma.reset_index(), on="date", how="left")

    for c in CAT_FEATURES:
        out[c] = out[c].astype("category")
    out["quake_period"] = out["date"].between(QUAKE_START, QUAKE_END).astype("int8")
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling means of log1p(sales); windows end before the target day.
    log1p(satış) lag ve rolling ortalamaları; pencereler hedef günden önce biter.
    """
    y = np.log1p(df["sales"])
    g = y.groupby([df["store_nbr"], df["family"]], observed=True)
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(1)
    g1 = shifted.groupby([df["store_nbr"], df["family"]], observed=True)
    for w in ROLL_WINDOWS:
        # Built-in grouped rolling is much faster than transform(lambda)
        # Yerleşik gruplu rolling, transform(lambda)'dan çok daha hızlı
        r = g1.rolling(w, min_periods=w).mean()
        df[f"rmean_{w}"] = r.reset_index(level=[0, 1], drop=True).reindex(df.index)
    # Mean of the same weekday over the last 4 weeks
    # Son 4 haftada aynı haftanın gününün ortalaması
    df["dow_mean_4w"] = sum(g.shift(k) for k in [7, 14, 21, 28]) / 4
    return df
