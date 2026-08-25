import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse explicit boolean environment flags with a fail-safe default."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_tuple(name: str, default: str) -> tuple[int, ...]:
    """Return the validated fixed horizon contract or its safe fallback."""
    fallback = tuple(int(item) for item in default.split(","))
    raw = os.getenv(name, default)
    try:
        values = tuple(dict.fromkeys(
            int(item.strip())
            for item in raw.split(",")
            if item.strip() and int(item.strip()) > 0
        ))
    except (TypeError, ValueError):
        values = ()
    if values == fallback:
        return values
    return fallback


@dataclass(frozen=True)
class Config:
    app_version: str = os.getenv(
        "MARKET_ANOMALY_VERSION",
        "2.2.0",
    )

    model_version: str = os.getenv(
        "MARKET_ANOMALY_MODEL_VERSION",
        "ma-core-2.2.0",
    )

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

    backup_dir: str = os.getenv(
        "MARKET_ANOMALY_BACKUP_DIR",
        "data/backups",
    )

    backup_retention: int = int(os.getenv(
        "MARKET_ANOMALY_BACKUP_RETENTION",
        "7",
    ))

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
        "eodhd",
    ).strip().lower()

    alpha_vantage_api_key: str = os.getenv(
        "ALPHA_VANTAGE_API_KEY",
        "",
    )

    data_mode: str = os.getenv(
        "MARKET_ANOMALY_DATA_MODE",
        "live",
    ).strip().lower()

    api_key: str = os.getenv(
        "MARKET_ANOMALY_API_KEY",
        "",
    )

    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT",
        "MarketAnomaly your-email@example.com",
    )

    # La cache storica non deve vivere per sempre: era la causa principale
    # dei prezzi vecchi mostrati dalla versione precedente.
    daily_cache_ttl_minutes: int = int(os.getenv(
        "MARKET_ANOMALY_DAILY_CACHE_TTL_MINUTES",
        "30",
    ))

    live_quote_ttl_seconds: int = int(os.getenv(
        "MARKET_ANOMALY_LIVE_QUOTE_TTL_SECONDS",
        "60",
    ))

    max_price_age_hours: int = int(os.getenv(
        "MARKET_ANOMALY_MAX_PRICE_AGE_HOURS",
        "72",
    ))

    screener_exchanges: str = os.getenv(
        "MARKET_ANOMALY_EXCHANGES",
        (
            "us,lse,to,pa,xetra,mi,sw,as,br,mc,ls,st,co,he,ol,"
            "tse,hk,au,jse"
        ),
    )

    light_universe_limit: int = int(os.getenv(
        "MARKET_ANOMALY_LIGHT_UNIVERSE_LIMIT",
        "10000",
    ))

    deep_candidate_limit: int = int(os.getenv(
        "MARKET_ANOMALY_DEEP_CANDIDATE_LIMIT",
        "200",
    ))

    minimum_home_confidence: float = float(os.getenv(
        "MARKET_ANOMALY_MIN_HOME_CONFIDENCE",
        "25",
    ))

    minimum_home_anomaly: float = float(os.getenv(
        "MARKET_ANOMALY_MIN_HOME_ANOMALY",
        "20",
    ))

    screener_max_requests: int = int(os.getenv(
        "MARKET_ANOMALY_SCREENER_MAX_REQUESTS",
        "25",
    ))

    provider_retry_count: int = int(os.getenv(
        "MARKET_ANOMALY_PROVIDER_RETRY_COUNT",
        "2",
    ))

    # Il backfill storico è un lavoro costoso e scrive un dataset di ricerca.
    # Rimane spento finché l'operatore non lo abilita esplicitamente e non
    # viene mai avviato automaticamente all'import o allo startup dell'API.
    historical_backfill_enabled: bool = _env_bool(
        "MARKET_ANOMALY_HISTORICAL_BACKFILL_ENABLED",
        False,
    )

    historical_backfill_default_years: int = max(1, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_BACKFILL_DEFAULT_YEARS",
        "10",
    )))

    historical_backfill_max_years: int = max(1, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_BACKFILL_MAX_YEARS",
        "15",
    )))

    historical_backfill_default_symbol_limit: int = max(1, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_BACKFILL_DEFAULT_SYMBOL_LIMIT",
        "250",
    )))

    historical_backfill_max_symbols: int = max(1, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_BACKFILL_MAX_SYMBOLS",
        "10000",
    )))

    historical_learning_horizons: tuple[int, ...] = _env_int_tuple(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_HORIZONS",
        "1,3,7,30,90,180",
    )

    historical_learning_baseline_sessions: int = max(20, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_BASELINE_SESSIONS",
        "60",
    )))

    historical_learning_minimum_history_sessions: int = max(60, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_MINIMUM_HISTORY_SESSIONS",
        "252",
    )))

    historical_learning_downside_threshold_pct: float = min(-0.01, float(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_DOWNSIDE_THRESHOLD_PCT",
        "-5.0",
    )))

    historical_learning_upside_threshold_pct: float = max(0.01, float(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_UPSIDE_THRESHOLD_PCT",
        "5.0",
    )))

    historical_learning_zscore_threshold: float = max(0.1, float(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_ZSCORE_THRESHOLD",
        "2.0",
    )))

    historical_learning_cooldown_sessions: int = max(0, int(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_COOLDOWN_SESSIONS",
        "5",
    )))

    historical_learning_recovery_tolerance_pct: float = max(0.0, float(os.getenv(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_RECOVERY_TOLERANCE_PCT",
        "0.0",
    )))

    historical_learning_require_all_horizons: bool = _env_bool(
        "MARKET_ANOMALY_HISTORICAL_LEARNING_REQUIRE_ALL_HORIZONS",
        True,
    )


    market_tension_refresh_hours: int = max(1, int(os.getenv(
        "MARKET_ANOMALY_TENSION_REFRESH_HOURS",
        "12",
    )))

    market_tension_sample_per_exchange: int = max(1, min(5, int(os.getenv(
        "MARKET_ANOMALY_TENSION_SAMPLE_PER_EXCHANGE",
        "5",
    ))))

    legal_terms_version: str = os.getenv(
        "MARKET_ANOMALY_TERMS_VERSION",
        "2026-08-25-v1",
    )

    legal_privacy_version: str = os.getenv(
        "MARKET_ANOMALY_PRIVACY_VERSION",
        "2026-08-25-v1",
    )

    legal_operator_name: str = os.getenv(
        "MARKET_ANOMALY_OPERATOR_NAME",
        "TIMONE TRASLOCHI E SERVIZI DI HELT ANTONI",
    )

    legal_operator_vat: str = os.getenv(
        "MARKET_ANOMALY_OPERATOR_VAT",
        "IT03132390216",
    )

    legal_operator_address: str = os.getenv(
        "MARKET_ANOMALY_OPERATOR_ADDRESS",
        "Via San Giacomo 53/A-1, 39055 Laives (BZ), Italia",
    )

    legal_privacy_contact: str = os.getenv(
        "MARKET_ANOMALY_PRIVACY_CONTACT",
        "antonilavoro@pec.it",
    )


CONFIG = Config()
