import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    app_name: str = os.getenv(
        "MARKET_ANOMALY_APP_NAME",
        "Market Anomaly",
    )

    data_dir: str = os.getenv(
        "MARKET_ANOMALY_DATA_DIR",
        "data",
    )

    db_path: str = os.getenv(
        "MARKET_ANOMALY_DB",
        "data/market_anomaly.db",
    )

    price_cache_dir: str = os.getenv(
        "MARKET_ANOMALY_PRICE_CACHE_DIR",
        "data/price_cache",
    )

    twelve_data_api_key: str = os.getenv(
        "TWELVE_DATA_API_KEY",
        "",
    )

    eodhd_api_key: str = os.getenv(
        "EODHD_API_KEY",
        "",
    )

    market_data_provider: str = os.getenv(
        "MARKET_ANOMALY_PROVIDER",
        "twelve_data",
    ).strip().lower()

    alpha_vantage_api_key: str = os.getenv(
        "ALPHA_VANTAGE_API_KEY",
        "",
    )

    data_mode: str = os.getenv(
        "MARKET_ANOMALY_DATA_MODE",
        "demo",
    ).strip().lower()

    api_key: str = os.getenv(
        "MARKET_ANOMALY_API_KEY",
        "",
    )

    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT",
        "MarketAnomaly your-email@example.com",
    )


CONFIG = Config()
