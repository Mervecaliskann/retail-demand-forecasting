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

# Oil prices for future days are unknown at forecast time: only use values >= 16 days old
# Gelecek günlerin petrol fiyatı tahmin anında bilinmez: sadece en az 16 gün eski değerler
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
    """One row per (date, store_nbr) with holiday flags.
    Her (tarih, mağaza) için bir satır, tatil bayraklarıyla.

    Rules / Kurallar:
    - transferred=True: the official date is a normal day, so drop it.
      transferred=True: resmi tarih normal bir gün, o yüzden at.
    - Transfer: the day the holiday is really celebrated, so count it as a holiday.
      Transfer: tatilin gerçekten kutlandığı gün, tatil say.
    - Bridge: extra day off. Work Day: weekend day that becomes a working day.
      Bridge: ek izin günü. Work Day: iş gününe dönen hafta sonu günü.
    - Earthquake events are not used here (handled by excluding those weeks).
      Deprem olayları burada kullanılmıyor (o haftalar eğitimden çıkarılıyor).
    """
    h = holidays[~holidays["transferred"]].copy()
    h = h[~h["description"].str.contains("Terremoto")]

    # Match each holiday to stores: national = all, regional = state, local = city
    # Her tatili mağazalarla eşle: ulusal = hepsi, bölgesel = eyalet, yerel = şehir
    s = stores[["store_nbr", "city", "state"]]
    nat = h[h["locale"] == "National"].merge(s[["store_nbr"]], how="cross")
    reg = h[h["locale"] == "Regional"].merge(s, left_on="locale_name", right_on="state")
    loc = h[h["locale"] == "Local"].merge(s, left_on="locale_name", right_on="city")
    allh = pd.concat([nat, reg, loc], ignore_index=True)
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


def add_exogenous(df: pd.DataFrame) -> pd.DataFrame:
    """Features that do not depend on sales: calendar, stores, holidays, promo, oil.
    Satıştan bağımsız özellikler: takvim, mağaza, tatil, promosyon, petrol.
    """
    stores = data.load_stores().rename(columns={"type": "store_type"})
    out = df.merge(stores[["store_nbr", "store_type", "cluster"]], on="store_nbr", how="left")

    # Calendar / Takvim
    d = out["date"].dt
    out["dayofweek"], out["day"], out["month"], out["dayofyear"] = d.dayofweek, d.day, d.month, d.dayofyear

    # Public-sector paydays: the 15th and the last day of the month
    # Kamu maaş günleri: ayın 15'i ve ayın son günü
    out["is_payday"] = ((d.day == 15) | d.is_month_end).astype("int8")
    out["days_since_payday"] = np.where(d.day >= 15, d.day - 15, d.day)

    # Holidays / Tatiller
    hol = holiday_table(data.load_holidays(), data.load_stores())
    out = out.merge(hol, on=["date", "store_nbr"], how="left")
    flags = ["hol_national", "hol_local", "is_bridge", "is_workday", "is_event"]
    out[flags] = out[flags].fillna(0).astype("int8")

    # Promotions are planned ahead, so same-day values are allowed
    # Promosyonlar önceden planlanır, aynı günün değeri kullanılabilir
    out["promo_share_family"] = out.groupby(["date", "family"], observed=True)["onpromotion"].transform(
        lambda x: (x > 0).mean())

    # Oil: 28-day mean, shifted by OIL_LAG days (only past prices)
    # Petrol: 28 günlük ortalama, OIL_LAG gün kaydırılmış (sadece geçmiş fiyatlar)
    oil = data.load_oil().set_index("date")["dcoilwtico"]
    oil_ma = oil.rolling(28, min_periods=1).mean().shift(OIL_LAG).rename("oil_ma28")
    out = out.merge(oil_ma.reset_index(), on="date", how="left")

    for c in CAT_FEATURES:
        out[c] = out[c].astype("category")
    out["quake_period"] = out["date"].between(QUAKE_START, QUAKE_END).astype("int8")
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling means of log1p(sales). Assumes rows sorted by series and date.
    log1p(satış) üzerinde lag ve rolling ortalamalar. Satırlar seri ve tarihe göre sıralı olmalı.

    All windows end at least 1 day before the target day (no leakage).
    Tüm pencereler hedef günden en az 1 gün önce biter (sızıntı yok).
    """
    y = np.log1p(df["sales"])
    g = y.groupby([df["store_nbr"], df["family"]], observed=True)
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)
    shifted = g.shift(1)
    g1 = shifted.groupby([df["store_nbr"], df["family"]], observed=True)
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = g1.transform(lambda s: s.rolling(w, min_periods=w).mean())
    # Mean of the same weekday over the last 4 weeks
    # Son 4 haftada aynı haftanın gününün ortalaması
    df["dow_mean_4w"] = sum(g.shift(k) for k in [7, 14, 21, 28]) / 4
    return df
