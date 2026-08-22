
from __future__ import annotations
import pandas as pd
import numpy as np

CORE_COLUMNS = [
    "ticker","company","sector_etf",
    "active_from","active_to","delisting_return_pct",
    "universe_source"
]

def normalize_universe(df: pd.DataFrame, source="custom_csv") -> pd.DataFrame:
    x = df.copy()

    rename = {
        "symbol":"ticker",
        "name":"company",
        "ipoDate":"active_from",
        "delistingDate":"active_to",
        "ipo_date":"active_from",
        "delisting_date":"active_to",
    }
    for a,b in rename.items():
        if a in x.columns and b not in x.columns:
            x = x.rename(columns={a:b})

    if "ticker" not in x.columns:
        raise ValueError("Il file universo deve contenere una colonna ticker oppure symbol.")

    x["ticker"] = x["ticker"].astype(str).str.upper().str.strip()
    x = x[x["ticker"].ne("") & x["ticker"].ne("NAN")].copy()

    if "company" not in x.columns:
        x["company"] = x["ticker"]
    x["company"] = x["company"].fillna(x["ticker"]).astype(str)

    if "sector_etf" not in x.columns:
        x["sector_etf"] = "SPY"
    x["sector_etf"] = x["sector_etf"].fillna("SPY").astype(str).str.upper()

    if "active_from" not in x.columns:
        x["active_from"] = pd.NaT
    if "active_to" not in x.columns:
        x["active_to"] = pd.NaT

    x["active_from"] = pd.to_datetime(x["active_from"], errors="coerce")
    x["active_to"] = pd.to_datetime(x["active_to"], errors="coerce")

    if "delisting_return_pct" not in x.columns:
        x["delisting_return_pct"] = np.nan
    x["delisting_return_pct"] = pd.to_numeric(x["delisting_return_pct"], errors="coerce")

    if "universe_source" not in x.columns:
        x["universe_source"] = source
    x["universe_source"] = x["universe_source"].fillna(source).astype(str)

    # Neutral demo fundamentals when a custom universe is tested with synthetic prices.
    demo_defaults = {
        "demo_revenue_growth": 8.0,
        "demo_net_margin": 10.0,
        "demo_liab_assets": 0.65,
        "demo_fcf_margin": 10.0,
    }
    for c,v in demo_defaults.items():
        if c not in x.columns:
            x[c] = v
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(v)

    # One lifecycle row per ticker. If duplicates exist, keep the widest known lifecycle.
    x = x.sort_values(["ticker","active_from","active_to"], na_position="last")
    x = x.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return x

def active_snapshot(universe: pd.DataFrame, date) -> pd.DataFrame:
    x = normalize_universe(universe, source="snapshot")
    d = pd.Timestamp(date).normalize()
    start_ok = x["active_from"].isna() | (x["active_from"] <= d)
    end_ok = x["active_to"].isna() | (x["active_to"] >= d)
    return x[start_ok & end_ok].copy().reset_index(drop=True)

def universe_summary(universe: pd.DataFrame) -> dict:
    x = normalize_universe(universe)
    with_start = int(x["active_from"].notna().sum())
    delisted = int(x["active_to"].notna().sum())
    terminal = int(x["delisting_return_pct"].notna().sum())
    return {
        "symbols": int(len(x)),
        "lifecycle_coverage_pct": round(with_start / len(x) * 100, 1) if len(x) else 0.0,
        "delisted_symbols": delisted,
        "terminal_return_coverage_pct": round(terminal / max(delisted,1) * 100, 1) if delisted else 0.0,
        "sources": ", ".join(sorted(x["universe_source"].dropna().astype(str).unique())),
    }

def build_demo_historical_universe(n=120, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = ["XLK","XLY","XLC","XLF","XLV","XLI","XLP","XLE","XLU","XLRE"]
    rows = []
    start_base = pd.Timestamp("2011-01-01")

    # Stable large-cap-like survivors.
    core = [
        ("AAPL","Apple","XLK"),("MSFT","Microsoft","XLK"),("GOOGL","Alphabet","XLC"),
        ("AMZN","Amazon","XLY"),("META","Meta Platforms","XLC"),("NVDA","NVIDIA","XLK"),
        ("AMD","AMD","XLK"),("CRM","Salesforce","XLK"),("ADBE","Adobe","XLK"),
        ("NOW","ServiceNow","XLK"),("PYPL","PayPal","XLF"),("NKE","Nike","XLY"),
        ("DIS","Walt Disney","XLC"),("SHOP","Shopify","XLK"),("UBER","Uber","XLI"),
        ("TSLA","Tesla","XLY"),("INTC","Intel","XLK"),("NFLX","Netflix","XLC"),
        ("ORCL","Oracle","XLK"),("JPM","JPMorgan Chase","XLF"),
    ]
    for i,(tic,name,sector) in enumerate(core):
        rows.append({
            "ticker":tic,"company":name,"sector_etf":sector,
            "active_from":pd.Timestamp("2011-01-01") + pd.Timedelta(days=int(i*70)),
            "active_to":pd.NaT,"delisting_return_pct":np.nan,
            "universe_source":"demo_historical",
        })

    # Synthetic names make the demo capable of testing entry/exit from the universe.
    remaining = max(0, int(n)-len(rows))
    for i in range(remaining):
        tic = f"H{i+1:03d}"
        ipo_year = int(rng.integers(2011, 2023))
        ipo_month = int(rng.integers(1,13))
        ipo_day = int(rng.integers(1,25))
        active_from = pd.Timestamp(ipo_year,ipo_month,ipo_day)

        will_delist = bool(rng.random() < 0.28)
        active_to = pd.NaT
        delist_ret = np.nan
        if will_delist:
            min_end = active_from + pd.Timedelta(days=365*2)
            max_end = pd.Timestamp("2025-12-31")
            if min_end < max_end:
                span = max((max_end-min_end).days,1)
                active_to = min_end + pd.Timedelta(days=int(rng.integers(0,span)))
                # Delisting terminal events skew negative, but can include takeovers.
                if rng.random() < 0.20:
                    delist_ret = float(rng.uniform(5,35))
                else:
                    delist_ret = float(rng.uniform(-100,-25))

        rows.append({
            "ticker":tic,
            "company":f"Historical Demo {i+1:03d}",
            "sector_etf":sectors[i % len(sectors)],
            "active_from":active_from,
            "active_to":active_to,
            "delisting_return_pct":delist_ret,
            "universe_source":"demo_historical",
        })

    df = pd.DataFrame(rows)

    # Demo fundamental variety.
    df["demo_revenue_growth"] = rng.normal(9, 11, len(df)).clip(-30,55)
    df["demo_net_margin"] = rng.normal(11, 13, len(df)).clip(-35,55)
    df["demo_liab_assets"] = rng.normal(0.65,0.16,len(df)).clip(0.15,1.15)
    df["demo_fcf_margin"] = rng.normal(10,12,len(df)).clip(-35,50)
    return normalize_universe(df, source="demo_historical")
