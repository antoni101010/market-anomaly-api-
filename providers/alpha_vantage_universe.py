from __future__ import annotations
from io import StringIO
import requests
import pandas as pd
import numpy as np

from universe_manager import normalize_universe

class AlphaVantageUniverseProvider:
    """
    Uses Alpha Vantage LISTING_STATUS to build an equity lifecycle universe.

    It is an optional *universe metadata* source. Price history still comes from
    the selected market-data provider.
    """
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout=30):
        self.api_key = api_key
        self.timeout = timeout

    def listing_status(self, state="active", date=None) -> pd.DataFrame:
        params = {
            "function":"LISTING_STATUS",
            "state":state,
            "apikey":self.api_key,
        }
        if date:
            params["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")

        r = requests.get(self.BASE, params=params, timeout=self.timeout)
        r.raise_for_status()
        text = r.text.strip()

        if not text:
            raise RuntimeError("Risposta vuota da Alpha Vantage.")
        if text.startswith("{"):
            # Rate-limit / informational message.
            try:
                msg = r.json()
            except Exception:
                msg = {"message": text[:300]}
            raise RuntimeError(str(msg))

        df = pd.read_csv(StringIO(text))
        if "symbol" not in df.columns:
            raise RuntimeError("Formato LISTING_STATUS non riconosciuto.")
        return df

    def build_lifecycle_universe(self, include_etfs=False, exchanges=None) -> pd.DataFrame:
        active = self.listing_status("active")
        delisted = self.listing_status("delisted")
        x = pd.concat([active, delisted], ignore_index=True)

        if "assetType" in x.columns and not include_etfs:
            x = x[x["assetType"].astype(str).str.lower().eq("stock")].copy()

        if exchanges and "exchange" in x.columns:
            allowed = {str(e).upper() for e in exchanges}
            x = x[x["exchange"].astype(str).str.upper().isin(allowed)].copy()

        out = pd.DataFrame({
            "ticker":x["symbol"].astype(str).str.upper(),
            "company":x["name"] if "name" in x.columns else x["symbol"],
            "sector_etf":"SPY",  # LISTING_STATUS has no sector classification.
            "active_from":x["ipoDate"] if "ipoDate" in x.columns else pd.NaT,
            "active_to":x["delistingDate"] if "delistingDate" in x.columns else pd.NaT,
            "delisting_return_pct":np.nan,
            "universe_source":"alpha_vantage_listing_status",
        })

        # Keep exchange/assetType as useful metadata.
        if "exchange" in x.columns:
            out["exchange"] = x["exchange"].values
        if "assetType" in x.columns:
            out["asset_type"] = x["assetType"].values

        return normalize_universe(out, source="alpha_vantage_listing_status")
