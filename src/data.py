"""Load the Favorita (Kaggle Store Sales) CSV files.
Favorita (Kaggle Store Sales) CSV dosyalarını yükler.
"""
from pathlib import Path

import pandas as pd

# Project root and data folder
# Proje kök klasörü ve veri klasörü
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Small dtypes keep ~3M rows light in memory
    # Küçük veri tipleri ~3M satırı bellekte hafif tutar
    df = pd.read_csv(
        data_dir / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category",
               "sales": "float32", "onpromotion": "int32"},
    )
    return df.drop(columns="id")


def load_stores(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "stores.csv")


def load_oil(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Oil has gaps (weekends, missing days): reindex daily and forward-fill.
    # Petrol verisinde boşluklar var (hafta sonu, eksik günler): günlük yap ve ileri doldur.
    # Forward-fill only uses past values, so no leakage.
    # İleri doldurma sadece geçmiş değeri kullanır, sızıntı yok.
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    full = pd.date_range(oil["date"].min(), "2017-08-31", freq="D")
    oil = oil.set_index("date").reindex(full).rename_axis("date")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()  # bfill only for the very first day / sadece ilk gün için
    return oil.reset_index()


def load_holidays(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])


def load_transactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])


# Project scope: small enough for a laptop, big enough to be realistic
# Proje kapsamı: laptopta hızlı çalışacak kadar küçük, gerçekçi olacak kadar büyük
SCOPE_START = "2015-07-01"
N_FAMILIES = 10


def load_scope(data_dir: Path = DATA_DIR, start: str = SCOPE_START,
               n_families: int = N_FAMILIES) -> pd.DataFrame:
    """All 54 stores x top-N families by total sales, from `start` on.
    54 mağaza x toplam satışa göre ilk N ürün ailesi, `start` tarihinden itibaren.
    """
    tr = load_train(data_dir)
    top = tr.groupby("family", observed=True)["sales"].sum().nlargest(n_families).index
    df = tr[(tr["date"] >= start) & tr["family"].isin(top)].copy()
    df["family"] = df["family"].cat.remove_unused_categories()

    # Drop days before a store's first sale (store not open yet)
    # Mağazanın ilk satışından önceki günleri at (mağaza henüz açık değil)
    first_sale = df[df["sales"] > 0].groupby("store_nbr")["date"].min()
    df = df[df["date"] >= df["store_nbr"].map(first_sale)]

    # Dec 25 is missing (stores closed): add it back with 0 sales so lags stay aligned
    # 25 Aralık yok (mağazalar kapalı): lag'ler kaymasın diye 0 satışla geri ekle
    idx = pd.MultiIndex.from_product(
        [pd.date_range(df["date"].min(), df["date"].max()),
         df["store_nbr"].unique(), df["family"].cat.categories],
        names=["date", "store_nbr", "family"])
    df = (df.set_index(["date", "store_nbr", "family"]).reindex(idx)
            .reset_index())
    df = df[df["date"] >= df["store_nbr"].map(first_sale)]
    df["is_closed"] = df["sales"].isna().astype("int8")  # filled days / doldurulan günler
    df["sales"] = df["sales"].fillna(0).astype("float32")
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("int32")
    df["family"] = df["family"].astype("category")
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
