from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    app_name: str = os.getenv("MARKET_ANOMALY_APP_NAME", "Market Anomaly")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "MarketAnomaly research contact@example.com")
    default_market: str = os.getenv("DEFAULT_MARKET", "US")
    db_path: str = os.getenv("MARKET_ANOMALY_DB", "data/market_anomaly.db")
    price_cache_dir: str = os.getenv("PRICE_CACHE_DIR", "data/price_cache")
    commercial_mode: bool = os.getenv("COMMERCIAL_MODE", "0").lower() in {"1","true","yes"}

    # Server-side only: these must NEVER be embedded in the Android app.
    # The API reads them from the server environment and the app only ever
    # talks to our own API, never directly to Twelve Data / Alpha Vantage.
    twelve_data_api_key: str = os.getenv("TWELVE_DATA_API_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # "demo" = dati sintetici, nessuna API key necessaria, nessun costo.
    # "live" = usa Twelve Data + SEC EDGAR (richiede TWELVE_DATA_API_KEY).
    data_mode: str = os.getenv("MARKET_ANOMALY_DATA_MODE", "demo")

    # API auth: chiave semplice condivisa tra app e backend per la v1
    # (da sostituire con login utenti veri in una fase successiva).
    api_key: str = os.getenv("MARKET_ANOMALY_API_KEY", "")

CONFIG = AppConfig()
