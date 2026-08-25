"""
Layer di servizio: collega scanner, provider dati, storage e API.
"""

from __future__ import annotations

import math
import json
import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import CONFIG
from scanner import (
    scan_universe,
    build_light_universe,
    _market_benchmark,
    _currency_for,
    quote_matches_reference,
)
from providers.twelve_data import TwelveDataProvider
from providers.eodhd import EODHDProvider
from historical_learning import (
    HistoricalLearningConfig,
    historical_learning_stats,
    historical_learning_status,
    run_historical_backfill,
)

from storage import (
    save_signals,
    load_signals,
    save_latest_scan,
    load_latest_scan,
    save_latest_light_scan,
    load_latest_light_scan,
    add_to_watchlist,
    remove_from_watchlist,
    list_watchlist,
    is_in_watchlist,
    save_scan_state,
    load_scan_state,
    save_signal_snapshots,
    list_due_outcomes,
    save_outcome,
    save_user_feedback,
    learning_summary,
    begin_scan_run,
    finish_scan_run,
    diagnostics as storage_diagnostics,
    create_database_backup,
    load_snapshot_history,
    save_market_tension_snapshot,
    load_market_tension_snapshot,
    market_tension_history as storage_market_tension_history,
    save_legal_acceptance,
    delete_legal_installation_data,
)

from narrative import build_ticker_narrative
from market_tension import collect_market_tension


_provider_instance = None
_provider_lock = threading.Lock()


FALLBACK_LIVE_UNIVERSE = [
    {"ticker": "AAPL.US", "display_ticker": "AAPL", "company": "Apple", "sector_etf": "SPY"},
    {"ticker": "MSFT.US", "display_ticker": "MSFT", "company": "Microsoft", "sector_etf": "SPY"},
    {"ticker": "NVDA.US", "display_ticker": "NVDA", "company": "NVIDIA", "sector_etf": "SPY"},
    {"ticker": "AMZN.US", "display_ticker": "AMZN", "company": "Amazon", "sector_etf": "SPY"},
    {"ticker": "GOOGL.US", "display_ticker": "GOOGL", "company": "Alphabet", "sector_etf": "SPY"},
    {"ticker": "META.US", "display_ticker": "META", "company": "Meta Platforms", "sector_etf": "SPY"},
    {"ticker": "TSLA.US", "display_ticker": "TSLA", "company": "Tesla", "sector_etf": "SPY"},
    {"ticker": "CRM.US", "display_ticker": "CRM", "company": "Salesforce", "sector_etf": "SPY"},
]


def _fallback_live_universe(limit: int) -> pd.DataFrame:
    rows = list(FALLBACK_LIVE_UNIVERSE)
    known = {
        str(item["display_ticker"]).upper()
        for item in rows
    }
    universe_path = Path(__file__).resolve().parent / "universe.csv"

    if universe_path.exists():
        try:
            universe = pd.read_csv(universe_path)

            for _, item in universe.iterrows():
                ticker = str(item.get("ticker", "")).strip().upper()

                if not ticker or ticker in known:
                    continue

                known.add(ticker)
                rows.append({
                    "ticker": f"{ticker}.US",
                    "display_ticker": ticker,
                    "company": str(item.get("company", ticker)),
                    "sector_etf": str(item.get("sector_etf", "SPY")),
                })
        except Exception:
            pass

    frame = pd.DataFrame(rows).head(int(limit)).copy()
    if not frame.empty:
        sector_names = {
            "XLK": "Technology",
            "XLV": "Healthcare",
            "XLF": "Financial Services",
            "XLY": "Consumer Cyclical",
            "XLP": "Consumer Defensive",
            "XLC": "Communication Services",
            "XLI": "Industrials",
            "XLE": "Energy",
            "XLB": "Basic Materials",
            "XLRE": "Real Estate",
            "XLU": "Utilities",
            "SPY": "Diversified",
        }
        frame["benchmark_ticker"] = "SPY.US"
        frame["light_exchange"] = "US"
        frame["currency"] = "USD"
        frame["country"] = "United States"
        frame["asset_type"] = "Common Stock"
        frame["light_sector"] = frame["sector_etf"].map(
            lambda value: sector_names.get(
                str(value).upper().replace(".US", ""),
                "Diversified",
            )
        )
    return frame


def _market_provider():
    global _provider_instance

    with _provider_lock:
        if _provider_instance is not None:
            return _provider_instance

    if CONFIG.data_mode == "live":

        if CONFIG.market_data_provider == "eodhd":
            if not CONFIG.eodhd_api_key:
                raise RuntimeError(
                    "EODHD_API_KEY non è impostata sul server."
                )

            provider = EODHDProvider(
                CONFIG.eodhd_api_key,
                cache_dir=CONFIG.price_cache_dir,
                daily_cache_ttl_minutes=CONFIG.daily_cache_ttl_minutes,
                live_quote_ttl_seconds=CONFIG.live_quote_ttl_seconds,
                retry_count=CONFIG.provider_retry_count,
                screener_max_requests=CONFIG.screener_max_requests,
                bulk_cache_ttl_minutes=CONFIG.bulk_cache_ttl_minutes,
            )
            with _provider_lock:
                _provider_instance = provider
            return provider

        if CONFIG.market_data_provider == "twelve_data":
            if not CONFIG.twelve_data_api_key:
                raise RuntimeError(
                    "TWELVE_DATA_API_KEY non è impostata sul server."
                )

            provider = TwelveDataProvider(
                CONFIG.twelve_data_api_key,
                cache_dir=CONFIG.price_cache_dir,
                cache_ttl_minutes=CONFIG.daily_cache_ttl_minutes,
            )
            with _provider_lock:
                _provider_instance = provider
            return provider

        raise RuntimeError(
            "Provider non riconosciuto. "
            "Usa eodhd oppure twelve_data."
        )

    raise RuntimeError(
        "Market Anomaly 2.0 accetta solamente dati reali. "
        "Imposta MARKET_ANOMALY_DATA_MODE=live."
    )


def _include_sec() -> bool:
    return CONFIG.data_mode == "live"


def _clean_json_value(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


def _row_to_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    return {
        k: _clean_json_value(v)
        for k, v in d.items()
    }


def _finite_max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    value = float(values.max())
    return value if math.isfinite(value) else None


def _price_metadata(record: dict) -> dict:
    observed = record.get("price_observed_at")
    age_hours = None
    try:
        timestamp = pd.Timestamp(observed)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        age_hours = max(
            0.0,
            (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds() / 3600,
        )
    except Exception:
        pass

    source = str(record.get("price_source") or "unknown")
    validation = str(record.get("price_validation") or "unknown")

    if validation == "provider_conflict":
        status = "conflict"
    elif age_hours is None:
        status = "unknown"
    elif age_hours > CONFIG.max_price_age_hours:
        status = "stale"
    elif source == "eodhd_websocket_realtime":
        market_status = str(record.get("market_status") or "").lower()
        if market_status == "closed":
            status = "market_closed"
        elif market_status in {"extended-hours", "extended_hours", "extended"}:
            status = "extended_hours"
        else:
            status = "live"
    elif "live" in source:
        status = "delayed" if record.get("price_is_delayed", True) else "live"
    else:
        status = "last_close"

    return {
        "price_observed_at": observed,
        "price_source": source,
        "price_status": status,
        "price_age_hours": round(age_hours, 2) if age_hours is not None else None,
        "price_warning": record.get("price_warning"),
    }


def _refresh_record_quote(
    record: dict,
    provider=None,
    *,
    prefer_realtime: bool = False,
) -> dict:
    refreshed = dict(record)
    if CONFIG.data_mode != "live":
        return refreshed

    try:
        provider = provider or _market_provider()
        provider_ticker = str(
            refreshed.get("provider_ticker")
            or f"{refreshed.get('ticker', '')}.US"
        )
        quote = None
        if (
            prefer_realtime
            and provider_ticker.upper().endswith(".US")
            and hasattr(provider, "realtime_trade_quote")
        ):
            try:
                quote = provider.realtime_trade_quote(provider_ticker)
            except Exception as realtime_error:
                refreshed["realtime_warning"] = str(realtime_error)
        if quote is None:
            if not hasattr(provider, "latest_quote"):
                return refreshed
            quote = provider.latest_quote(provider_ticker)
        price = float(quote["price"])
        previous = float(refreshed.get("last_close") or 0)
        if not quote_matches_reference(quote, previous):
            refreshed["price_validation"] = "provider_conflict"
            refreshed["price_warning"] = (
                "Il nuovo prezzo non coincide con lo storico verificato."
            )
            return refreshed
        refreshed.update({
            "last_close": price,
            "price_observed_at": quote.get("observed_at"),
            "price_source": quote.get("source"),
            "price_is_delayed": quote.get("is_delayed", True),
            "previous_close": quote.get("previous_close"),
            "live_change_pct": quote.get("change_pct"),
            "market_status": quote.get("market_status"),
            "price_validation": "ok",
        })
    except Exception as error:
        refreshed.setdefault("price_warning", str(error))
    return refreshed


GLOBAL_SCAN_EXCHANGES = (
    # EODHD uses the composite US exchange code for whole-exchange Bulk EOD.
    # Venue metadata (NASDAQ/NYSE/AMEX/BATS) is recovered from the symbol list.
    "US", "LSE", "TO", "V", "PA", "XETRA", "F", "MI", "SW",
    "AS", "BR", "MC", "LS", "ST", "CO", "HE", "OL", "TSE",
    "HK", "AU", "JSE", "WAR",
)


def _bulk_exchange_code(value: str) -> str:
    code = str(value or "").strip().upper()
    aliases = {
        "NASDAQ": "US", "NYSE": "US", "AMEX": "US", "BATS": "US",
        "NYSE MKT": "US", "NYSE ARCA": "US", "AX": "AU",
    }
    return aliases.get(code, code)


def _global_scan_exchanges() -> tuple[str, ...]:
    """Keep the global core universe even when Render has older env values.

    Old deployments stored NASDAQ/NYSE/AMEX/BATS separately.  Those are valid
    venues but the EODHD whole-exchange Bulk endpoint expects ``US``.  Normalize
    them here so the scanner cannot silently lose the US universe (and titles
    such as APP) because of a stale environment variable.
    """
    configured = [
        _bulk_exchange_code(item)
        for item in str(CONFIG.screener_exchanges or "").split(",")
        if item.strip()
    ]
    return tuple(dict.fromkeys([*GLOBAL_SCAN_EXCHANGES, *configured]))


def _select_deep_candidates(light_universe: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Global ranking plus geographic coverage.

    Most slots go to the strongest Light anomalies worldwide; the remaining
    slots guarantee that smaller exchanges are not erased by the US universe.
    """
    if light_universe is None or light_universe.empty:
        return pd.DataFrame()
    frame = light_universe.copy()
    if "light_anomaly_score" not in frame.columns:
        frame["light_anomaly_score"] = 0.0
    frame["light_anomaly_score"] = pd.to_numeric(
        frame["light_anomaly_score"], errors="coerce"
    ).fillna(0.0)
    frame = frame.sort_values(
        ["light_anomaly_score", "light_market_cap_usd"],
        ascending=[False, False],
        na_position="last",
    ).drop_duplicates(subset=["ticker"], keep="first")

    requested = max(1, min(int(limit), max(300, int(CONFIG.deep_candidate_limit))))
    global_slots = max(1, int(round(requested * 0.70)))
    selected = frame.head(global_slots).copy()

    exchange_col = "source_exchange" if "source_exchange" in frame.columns else "light_exchange"
    exchanges = [
        value for value in frame.get(exchange_col, pd.Series(dtype=str)).dropna().astype(str).unique()
        if value
    ]
    remaining_slots = max(0, requested - len(selected))
    if remaining_slots and exchanges:
        per_exchange = max(1, math.ceil(remaining_slots / len(exchanges)))
        coverage_rows = []
        for exchange in exchanges:
            group = frame[frame[exchange_col].astype(str) == str(exchange)].head(per_exchange)
            if not group.empty:
                coverage_rows.append(group)
        if coverage_rows:
            selected = pd.concat([selected, *coverage_rows], ignore_index=True)

    selected = selected.drop_duplicates(subset=["ticker"], keep="first")

    # Geographic coverage can duplicate names already taken by the global top.
    # Fill any gap from the remaining globally ranked universe so the Deep
    # budget is actually used (e.g. 300 means 300 whenever 300 are available).
    if len(selected) < requested:
        missing = requested - len(selected)
        remaining = frame[~frame["ticker"].isin(selected["ticker"])].head(missing)
        if not remaining.empty:
            selected = pd.concat([selected, remaining], ignore_index=True)

    selected = (
        selected.drop_duplicates(subset=["ticker"], keep="first")
        .sort_values(["light_anomaly_score", "light_market_cap_usd"], ascending=[False, False], na_position="last")
        .head(requested)
        .reset_index(drop=True)
    )
    return selected


def _build_live_universe(provider, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan the broad eligible global universe, then select Deep candidates."""
    exchanges = _global_scan_exchanges()

    light_universe = build_light_universe(
        market_provider=provider,
        exchanges=exchanges,
        # Main scanner keeps the previously agreed micro-cap exclusion.
        min_avg_volume=50_000,
        min_price=1.0,
        min_market_cap=500_000_000,
        max_rows=max(50_000, int(CONFIG.light_universe_limit)),
    )

    if light_universe is None:
        raise RuntimeError("Il provider selezionato non supporta il Light Scanner globale.")

    if light_universe.empty:
        fallback = _fallback_live_universe(limit)
        fallback["light_anomaly_score"] = 0.0
        fallback["light_scanner_mode"] = "fallback_without_bulk"
        fallback["light_scanner_note"] = (
            "Bulk globale non disponibile: universo di emergenza usato solo per diagnostica."
        )
        return fallback, fallback.head(min(int(limit), CONFIG.deep_candidate_limit)).copy()

    light_universe = light_universe.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    light_universe["light_universe_rank"] = range(1, len(light_universe) + 1)
    stats = getattr(provider, "last_bulk_stats", {}) or {}
    light_universe["light_universe_scanned"] = int(len(light_universe))
    light_universe["light_raw_rows"] = int(stats.get("raw_rows") or len(light_universe))
    light_universe["bulk_exchanges_scanned"] = ",".join(stats.get("exchanges_scanned") or [])
    light_universe["bulk_estimated_api_calls"] = int(stats.get("estimated_api_calls") or 0)
    light_universe["light_scanner_mode"] = "eodhd_bulk_global"

    deep = _select_deep_candidates(light_universe, limit)
    return light_universe, deep


def _run_scan_core(
    limit: int = 300,
    catalyst_top_n: int = 120,
    *,
    run_id: str,
) -> dict:

    if CONFIG.data_mode != "live":
        raise RuntimeError("La modalità demo è disattivata: servono dati reali.")

    provider = _market_provider()
    if CONFIG.data_mode == "live":
        light_universe, live_u = _build_live_universe(
            provider,
            limit=limit,
        )

        if light_universe.empty or live_u.empty:
            empty = pd.DataFrame()
            save_latest_scan(empty, market_mode="live")
            save_latest_light_scan(
                empty,
                market_mode="live",
                universe_scanned=0,
                anomaly_candidates=0,
            )
            result = {
                "run_id": run_id,
                "scanned": 0,
                "valid": 0,
                "failed": 0,
                "data_mode": "live",
                "provider": CONFIG.market_data_provider,
                "universe_scanned": 0,
                "light_candidates": 0,
                "deep_selected": 0,
                "scanner_mode": "eodhd_bulk_global",
            }
            finish_scan_run(run_id, {**result, "status": "done"})
            return result

        light_scores = pd.to_numeric(
            light_universe.get("light_anomaly_score", pd.Series(0, index=light_universe.index)),
            errors="coerce",
        ).fillna(0.0)
        anomaly_count = int((light_scores >= float(CONFIG.light_anomaly_threshold)).sum())
        # The full universe is scanned, while a broad Light catalogue is kept
        # locally so users can browse normal/moderate anomalies without forcing
        # thousands of expensive Deep analyses. 20k compact rows are small
        # enough for SQLite but wide enough to avoid the old 25/50-title effect.
        persisted_light = light_universe[light_scores >= 5.0].head(20_000).copy()
        save_latest_light_scan(
            persisted_light,
            market_mode="live",
            universe_scanned=int(len(light_universe)),
            anomaly_candidates=anomaly_count,
        )

    effective_catalyst_n = min(
        max(1, int(catalyst_top_n)),
        int(CONFIG.catalyst_candidate_limit),
        int(len(live_u)),
    )
    df = scan_universe(
        live_u,
        provider,
        include_sec=_include_sec(),
        catalyst_top_n=effective_catalyst_n,
    )
    if not df.empty:
        df["model_version"] = CONFIG.model_version
        df["data_provider"] = CONFIG.market_data_provider
        df["global_universe_scanned"] = int(len(light_universe))
        df["light_anomaly_candidates"] = int(anomaly_count)
        df["deep_selected"] = int(len(live_u))

    save_latest_scan(
        df,
        market_mode=CONFIG.data_mode,
    )

    valid = (
        df[df["error"].isna()].copy()
        if "error" in df.columns
        else df
    )

    if not valid.empty:
        save_signals(valid)
        save_signal_snapshots(
            valid,
            model_version=CONFIG.model_version,
        )

    result = {
        "run_id": run_id,
        "scanned": int(len(df)),
        "valid": int(len(valid)),
        "failed": int(len(df) - len(valid)),
        "data_mode": CONFIG.data_mode,
        "provider": CONFIG.market_data_provider,
        "universe_scanned": int(len(light_universe)),
        "light_candidates": int(anomaly_count),
        "deep_selected": int(len(live_u)),
        "scanner_mode": "eodhd_bulk_global",
        "universe_target": int(CONFIG.light_universe_limit),
        "bulk_estimated_api_calls": int(
            live_u.iloc[0].get("bulk_estimated_api_calls", 0) if len(live_u) else 0
        ),
        "bulk_exchanges_scanned": str(
            live_u.iloc[0].get("bulk_exchanges_scanned", "") if len(live_u) else ""
        ),
        "catalyst_analyzed": int(effective_catalyst_n),
    }

    finish_scan_run(run_id, {**result, "status": "done"})
    try:
        result["backup_path"] = create_database_backup()
    except Exception as backup_error:
        result["backup_warning"] = str(backup_error)
    return result


def run_scan(
    limit: int = 300,
    catalyst_top_n: int = 120,
) -> dict:
    """Esegue una scansione e chiude sempre il relativo audit operativo."""
    run_id = uuid.uuid4().hex
    begin_scan_run(run_id, CONFIG.market_data_provider, limit)

    try:
        return _run_scan_core(
            limit=limit,
            catalyst_top_n=catalyst_top_n,
            run_id=run_id,
        )
    except Exception as error:
        finish_scan_run(
            run_id,
            {
                "status": "error",
                "message": str(error),
                "scanned": 0,
                "valid": 0,
                "failed": 0,
            },
        )
        raise


_market_tension_lock = threading.Lock()
_market_tension_refresh_running = False
_market_tension_last_error: str | None = None


def _market_tension_is_fresh(payload: dict | None) -> bool:
    if not payload or not payload.get("observed_at"):
        return False
    try:
        observed = pd.Timestamp(payload["observed_at"])
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        age_hours = (
            pd.Timestamp.now(tz="UTC") - observed
        ).total_seconds() / 3600.0
        return age_hours < float(CONFIG.market_tension_refresh_hours)
    except Exception:
        return False


def get_market_tension() -> dict:
    payload = load_market_tension_snapshot()
    if payload is None:
        return {
            "observed_at": None,
            "status": "unavailable",
            "score": None,
            "level": "Non disponibile",
            "valuation_pressure": None,
            "price_euphoria": None,
            "fragility": None,
            "coverage_pct": 0.0,
            "source": CONFIG.market_data_provider,
            "methodology_version": "market-tension-3.0",
            "explanation": (
                "Il primo indicatore globale non e ancora stato calcolato. "
                "La scansione automatica lo aggiornera usando un campione neutrale multi-mercato."
            ),
            "last_error": _market_tension_last_error,
            "data_delay_note": (
                "Indicatore basato su dati EOD e fondamentali: non e un dato real-time."
            ),
            "historical_warning": (
                "Relazioni storiche, confronti e backtest non garantiscono risultati futuri."
            ),
            "not_investment_advice": True,
        }
    if _market_tension_last_error and payload.get("status") in {"partial", "unavailable"}:
        payload = {**payload, "last_error": _market_tension_last_error}
    return payload


def refresh_market_tension(*, force: bool = False) -> dict:
    global _market_tension_last_error
    cached = load_market_tension_snapshot()
    if not force and _market_tension_is_fresh(cached):
        return {**cached, "refresh": "cached"}

    provider = _market_provider()
    exchanges = _global_scan_exchanges()
    result = collect_market_tension(
        provider,
        exchanges=exchanges,
        sample_per_exchange=max(15, int(CONFIG.market_tension_sample_per_exchange)),
    )
    save_market_tension_snapshot(result)
    _market_tension_last_error = None
    return {**result, "refresh": "computed"}


def _run_market_tension_refresh() -> None:
    global _market_tension_refresh_running, _market_tension_last_error
    try:
        refresh_market_tension(force=False)
    except Exception as error:
        # Non blocca lo scanner, ma l'errore resta visibile alla diagnostica/UI.
        _market_tension_last_error = str(error)
    finally:
        with _market_tension_lock:
            _market_tension_refresh_running = False


def start_market_tension_background(*, force: bool = False) -> dict:
    global _market_tension_refresh_running
    cached = load_market_tension_snapshot()
    if not force and _market_tension_is_fresh(cached):
        return {"ok": True, "status": "cached", "payload": cached}
    with _market_tension_lock:
        if _market_tension_refresh_running:
            return {"ok": False, "status": "running"}
        _market_tension_refresh_running = True
    thread = threading.Thread(target=_run_market_tension_refresh, daemon=True)
    thread.start()
    return {"ok": True, "status": "started"}


def get_market_tension_history(limit: int = 90) -> list[dict]:
    return storage_market_tension_history(limit=limit)


def get_legal_current() -> dict:
    return {
        "terms_version": CONFIG.legal_terms_version,
        "privacy_version": CONFIG.legal_privacy_version,
        "operator": {
            "name": CONFIG.legal_operator_name,
            "vat": CONFIG.legal_operator_vat,
            "address": CONFIG.legal_operator_address,
            "privacy_contact": CONFIG.legal_privacy_contact,
        },
        "product_positioning": "statistical_research_tool",
        "personalized_advice": False,
        "trading_execution": False,
        "portfolio_management": False,
    }


def record_legal_acceptance(
    installation_id: str,
    *,
    terms_version: str,
    privacy_version: str,
    app_version: str = "",
    platform: str = "",
    terms_accepted: bool = True,
    privacy_notice_acknowledged: bool = True,
) -> dict:
    if terms_version != CONFIG.legal_terms_version:
        raise ValueError("Versione dei Termini non corrente.")
    if privacy_version != CONFIG.legal_privacy_version:
        raise ValueError("Versione dell'informativa privacy non corrente.")
    if not terms_accepted:
        raise ValueError("I Termini devono essere accettati per usare il servizio.")
    return save_legal_acceptance(
        installation_id,
        terms_version=terms_version,
        privacy_version=privacy_version,
        app_version=app_version,
        platform=platform,
        terms_accepted=terms_accepted,
        privacy_notice_acknowledged=privacy_notice_acknowledged,
    )


def delete_installation_data(installation_id: str) -> dict:
    deleted = delete_legal_installation_data(installation_id)
    return {"ok": True, "deleted_records": int(deleted)}


_scan_state = {
    "status": "idle",
    "message": "",
    "limit": 0,
    "started_at": None,
    "finished_at": None,
}

_scan_lock = threading.Lock()


_persisted_state = load_scan_state()

if _persisted_state:
    _scan_state.update(
        _persisted_state
    )

    if _scan_state["status"] == "running":
        _scan_state.update(
            status="error",
            message=(
                "La scansione è stata interrotta "
                "dal riavvio del server. Riprova."
            ),
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        save_scan_state(
            _scan_state
        )


def get_scan_status() -> dict:
    with _scan_lock:
        return dict(
            _scan_state
        )


def _set_scan_state(**kwargs):
    with _scan_lock:
        _scan_state.update(
            kwargs
        )

        save_scan_state(
            _scan_state
        )


def _run_scan_thread(
    limit: int,
    catalyst_top_n: int,
):
    try:
        result = run_scan(
            limit=limit,
            catalyst_top_n=catalyst_top_n,
        )

        _set_scan_state(
            status="done",
            message=(
                f"Completata: universo globale {result.get('universe_scanned', 0):,} titoli; "
                f"{result.get('light_candidates', 0):,} movimenti Light rilevanti; "
                f"{result['valid']} analisi Deep valide su {result['scanned']}. "
                f"Modalità: {result['scanner_mode']}."
            ),
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        # Aggiorna in parallelo l'indicatore globale senza bloccare la scansione.
        start_market_tension_background(force=False)

    except Exception as e:
        _set_scan_state(
            status="error",
            message=str(e),
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )


def start_scan_background(
    limit: int = 300,
    catalyst_top_n: int = 120,
) -> dict:

    with _scan_lock:
        if _scan_state["status"] == "running":
            return {
                "ok": False,
                "message": (
                    "Una scansione è già in corso."
                ),
            }

        _scan_state.update(
            status="running",
            message="Scansione avviata...",
            limit=int(limit),
            started_at=datetime.now(
                timezone.utc
            ).isoformat(),
            finished_at=None,
        )

        save_scan_state(
            _scan_state
        )

    thread = threading.Thread(
        target=_run_scan_thread,
        args=(
            limit,
            catalyst_top_n,
        ),
        daemon=True,
    )

    thread.start()

    return {
        "ok": True,
        "message": (
            "Scansione avviata in background."
        ),
    }


_historical_backfill_state = {
    "status": "idle",
    "message": "",
    "job_id": None,
    "run_id": None,
    "run_key": None,
    "started_at": None,
    "finished_at": None,
    "years": None,
    "as_of": None,
    "symbols_total": 0,
    "primary_event_side": "downside",
    "automatic_production_weight_changes": False,
}

_historical_backfill_lock = threading.Lock()


def _historical_learning_config() -> HistoricalLearningConfig:
    """Build the research configuration without changing production weights."""
    return HistoricalLearningConfig(
        horizons=CONFIG.historical_learning_horizons,
        baseline_sessions=CONFIG.historical_learning_baseline_sessions,
        minimum_history_sessions=(
            CONFIG.historical_learning_minimum_history_sessions
        ),
        downside_return_threshold_pct=(
            CONFIG.historical_learning_downside_threshold_pct
        ),
        upside_return_threshold_pct=(
            CONFIG.historical_learning_upside_threshold_pct
        ),
        zscore_threshold=CONFIG.historical_learning_zscore_threshold,
        cooldown_sessions=CONFIG.historical_learning_cooldown_sessions,
        recovery_tolerance_pct=(
            CONFIG.historical_learning_recovery_tolerance_pct
        ),
        require_all_horizons=(
            CONFIG.historical_learning_require_all_horizons
        ),
        primary_event_side="downside",
        price_adjustment="all",
    )


def _normalize_backfill_as_of(value: str | None) -> str:
    if value is None or not str(value).strip():
        return datetime.now(timezone.utc).date().isoformat()
    try:
        timestamp = pd.Timestamp(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError("as_of non valido: usa YYYY-MM-DD.") from error
    if pd.isna(timestamp):
        raise ValueError("as_of non valido: usa YYYY-MM-DD.")
    as_of = timestamp.date()
    if as_of > datetime.now(timezone.utc).date():
        raise ValueError("as_of non può essere nel futuro.")
    return as_of.isoformat()


def _historical_backfill_universe(
    tickers: list[str] | None,
    *,
    limit: int,
) -> list[dict]:
    """Create a bounded neutral universe; never reuse the current loser screen."""
    _, _, latest_records = load_latest_scan()
    latest_by_ticker = {
        str(item.get("ticker") or "").strip().upper(): item
        for item in latest_records
        if str(item.get("ticker") or "").strip()
    }
    local_rows = _local_search_rows()
    local_by_ticker = {
        str(item.get("ticker") or "").strip().upper(): item
        for item in local_rows
        if str(item.get("ticker") or "").strip()
    }

    requested = []
    discovered_by_provider = {}
    for raw in tickers or []:
        provider_ticker = str(raw or "").strip().upper()
        if not provider_ticker:
            continue
        if len(provider_ticker) > 40 or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
            for character in provider_ticker
        ):
            raise ValueError(f"Ticker non valido: {provider_ticker!r}.")
        if provider_ticker not in requested:
            requested.append(provider_ticker)

    if not requested:
        # Il backfill deve partire da un universo neutrale, non dalla shortlist
        # dei titoli gia crollati. Con EODHD a pagamento enumera gli exchange
        # configurati; se il provider non lo consente usa la lista locale reale.
        try:
            provider = _market_provider()
            if hasattr(provider, "exchange_symbols"):
                for exchange in (
                    value.strip().upper()
                    for value in CONFIG.screener_exchanges.split(",")
                    if value.strip()
                ):
                    try:
                        discovered = provider.exchange_symbols(
                            exchange,
                            include_delisted=False,
                            common_stocks_only=True,
                        )
                    except Exception:
                        continue
                    for item in discovered:
                        provider_ticker = str(
                            item.get("provider_ticker") or ""
                        ).strip().upper()
                        if not provider_ticker or provider_ticker in discovered_by_provider:
                            continue
                        discovered_by_provider[provider_ticker] = item
                        requested.append(provider_ticker)
                        if len(requested) >= int(limit):
                            break
                    if len(requested) >= int(limit):
                        break
        except Exception:
            requested = []

    if not requested:
        requested = [
            str(item["provider_ticker"]).strip().upper()
            for item in local_rows
            if item.get("provider_ticker")
        ]

    rows = []
    seen = set()
    for requested_ticker in requested:
        display_ticker = requested_ticker.split(".", 1)[0]
        latest = latest_by_ticker.get(display_ticker, {})
        local = local_by_ticker.get(display_ticker, {})
        discovered = discovered_by_provider.get(requested_ticker, {})
        provider_ticker = str(
            latest.get("provider_ticker")
            or local.get("provider_ticker")
            or discovered.get("provider_ticker")
            or (
                requested_ticker
                if "." in requested_ticker
                else f"{requested_ticker}.US"
            )
        ).upper()
        if provider_ticker in seen:
            continue
        seen.add(provider_ticker)
        rows.append({
            "ticker": display_ticker,
            "provider_ticker": provider_ticker,
            "company": str(
                latest.get("company")
                or local.get("company")
                or discovered.get("company")
                or display_ticker
            ),
            "benchmark_ticker": str(
                latest.get("benchmark_ticker")
                or latest.get("sector_etf")
                or _market_benchmark(
                    str(
                        discovered.get("exchange")
                        or provider_ticker.rsplit(".", 1)[-1]
                        or "US"
                    )
                )
            ).upper(),
        })
        if len(rows) >= int(limit):
            break
    return rows


def _historical_run_key(
    universe: list[dict],
    *,
    years: int,
    as_of: str,
) -> str:
    payload = {
        "symbols": [item["provider_ticker"] for item in universe],
        "years": int(years),
        "as_of": as_of,
        "market_tension": get_market_tension(),
        "model_version": CONFIG.model_version,
        "horizons": list(CONFIG.historical_learning_horizons),
        "downside_threshold": (
            CONFIG.historical_learning_downside_threshold_pct
        ),
        "upside_threshold": (
            CONFIG.historical_learning_upside_threshold_pct
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"historical-{digest}"


def _set_historical_backfill_state(**values) -> None:
    with _historical_backfill_lock:
        _historical_backfill_state.update(values)


def _run_historical_backfill_thread(
    universe: list[dict],
    years: int,
    as_of: str,
    run_key: str,
    resume: bool,
) -> None:
    try:
        result = run_historical_backfill(
            universe,
            _market_provider(),
            years=years,
            as_of=as_of,
            model_version=CONFIG.model_version,
            run_key=run_key,
            config=_historical_learning_config(),
            resume=bool(resume),
        )
        result = dict(result or {})
        _set_historical_backfill_state(
            status=str(result.get("status") or "done"),
            message=(
                "Backfill storico completato; i risultati restano in ricerca "
                "finché non vengono validati e promossi manualmente."
            ),
            run_id=result.get("run_id"),
            finished_at=(
                result.get("finished_at")
                or datetime.now(timezone.utc).isoformat()
            ),
            symbols_total=int(
                result.get("symbols_total") or len(universe)
            ),
            result=result,
        )
    except Exception as error:
        _set_historical_backfill_state(
            status="error",
            message=str(error),
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(error),
        )


def start_historical_backfill_background(
    *,
    years: int | None = None,
    as_of: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Start one bounded backfill job; never auto-promote learned parameters."""
    if not CONFIG.historical_backfill_enabled:
        raise RuntimeError(
            "Backfill storico disabilitato sul server. Imposta "
            "MARKET_ANOMALY_HISTORICAL_BACKFILL_ENABLED=1 per abilitarlo."
        )

    requested_years = int(
        years
        if years is not None
        else CONFIG.historical_backfill_default_years
    )
    if not 1 <= requested_years <= CONFIG.historical_backfill_max_years:
        raise ValueError(
            "years deve essere compreso tra 1 e "
            f"{CONFIG.historical_backfill_max_years}."
        )

    requested_limit = int(
        limit
        if limit is not None
        else CONFIG.historical_backfill_default_symbol_limit
    )
    if not 1 <= requested_limit <= CONFIG.historical_backfill_max_symbols:
        raise ValueError(
            "limit deve essere compreso tra 1 e "
            f"{CONFIG.historical_backfill_max_symbols}."
        )
    if tickers and len(tickers) > CONFIG.historical_backfill_max_symbols:
        raise ValueError(
            "Troppi ticker: il massimo configurato è "
            f"{CONFIG.historical_backfill_max_symbols}."
        )

    normalized_as_of = _normalize_backfill_as_of(as_of)
    universe = _historical_backfill_universe(
        tickers,
        limit=requested_limit,
    )
    if not universe:
        raise ValueError("Nessun ticker valido disponibile per il backfill.")
    run_key = _historical_run_key(
        universe,
        years=requested_years,
        as_of=normalized_as_of,
    )
    job_id = uuid.uuid4().hex

    with _historical_backfill_lock:
        if _historical_backfill_state.get("status") == "running":
            return {
                "ok": False,
                "message": "Un backfill storico è già in corso.",
                "job_id": _historical_backfill_state.get("job_id"),
            }
        _historical_backfill_state.clear()
        _historical_backfill_state.update({
            "status": "running",
            "message": "Backfill storico avviato...",
            "job_id": job_id,
            "run_id": None,
            "run_key": run_key,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "years": requested_years,
            "as_of": normalized_as_of,
            "symbols_total": len(universe),
            "primary_event_side": "downside",
            "automatic_production_weight_changes": False,
        })

    thread = threading.Thread(
        target=_run_historical_backfill_thread,
        args=(
            universe,
            requested_years,
            normalized_as_of,
            run_key,
            bool(resume),
        ),
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "message": "Backfill storico avviato in background.",
        "job_id": job_id,
        "run_key": run_key,
        "symbols_total": len(universe),
        "years": requested_years,
        "as_of": normalized_as_of,
        "primary_event_side": "downside",
        "automatic_production_weight_changes": False,
    }


def get_historical_backfill_status() -> dict:
    """Return live process state enriched with the persisted checkpoint."""
    with _historical_backfill_lock:
        state = dict(_historical_backfill_state)
    try:
        persisted = historical_learning_status(state.get("run_id"))
    except Exception as error:
        persisted = {"status_read_error": str(error)}
    state["persisted"] = persisted or None
    state["enabled"] = bool(CONFIG.historical_backfill_enabled)
    state["primary_event_side"] = "downside"
    state["automatic_production_weight_changes"] = False
    return state


def get_historical_learning_stats() -> dict:
    """Expose downside and upside cohorts separately, never as one signal."""
    result = dict(historical_learning_stats() or {})
    directions = result.get("directions") or result.get("by_direction") or {}
    if isinstance(directions, list):
        directions = {
            str(item.get("event_side") or item.get("direction")): item
            for item in directions
            if isinstance(item, dict)
        }
    if not isinstance(directions, dict):
        directions = {}

    event_counts = result.get("events_by_side") or {}
    performance_rows = result.get("performance_by_side_and_horizon") or []
    for side in ("downside", "upside"):
        direction = dict(directions.get(side) or {})
        direction.setdefault("events", int(event_counts.get(side) or 0))
        direction.setdefault(
            "performance_by_horizon",
            [
                dict(item)
                for item in performance_rows
                if str(item.get("event_side") or "").lower() == side
            ],
        )
        directions[side] = direction

    result["directions"] = {
        "downside": dict(directions.get("downside") or {}),
        "upside": dict(directions.get("upside") or {}),
    }
    result["primary_event_side"] = "downside"
    result["automatic_production_weight_changes"] = False
    result["promotion_required"] = (
        "Backtest point-in-time, holdout, walk-forward e approvazione esplicita."
    )
    return result



def _light_classification(score: float | None) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if value >= 75:
        return "ANOMALIA LIGHT MOLTO FORTE"
    if value >= 60:
        return "ANOMALIA LIGHT FORTE"
    if value >= 40:
        return "ANOMALIA LIGHT MODERATA"
    return "ANOMALIA LIGHT NORMALE"


def _light_summary(record: dict) -> str:
    def finite(value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    score = finite(record.get("light_anomaly_score"))
    drawdown = finite(record.get("light_drawdown_250d_pct"))
    daily = finite(record.get("light_return_1d_pct"))
    volume_ratio = finite(record.get("light_volume_ratio"))
    parts = ["Screening Light globale"]
    if drawdown is not None:
        parts.append(f"ribasso dal massimo circa {abs(drawdown):.1f}%")
    if daily is not None and abs(daily) >= 2:
        parts.append(f"movimento giornaliero {daily:+.1f}%")
    if volume_ratio is not None and volume_ratio >= 1.5:
        parts.append(f"volume {volume_ratio:.1f}x la media recente")
    if score is not None:
        parts.append(f"score Light {score:.0f}/100")
    return ". ".join(parts) + ". Apri la scheda per eseguire l’analisi Deep completa."


def _prepare_light_dashboard_frame(
    records: list[dict],
    *,
    market: str,
    company_size: str,
    selected_sectors: set[str],
    min_anomaly: float,
    max_anomaly: float,
    min_drawdown_pct: float,
    min_average_volume: int,
    exclude_tickers: set[str],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        return frame

    display = frame.get("display_ticker", frame.get("ticker", pd.Series("", index=frame.index)))
    frame["display_ticker"] = display.astype(str).str.upper().str.split(".").str[0]
    frame["provider_ticker"] = frame.get("ticker", pd.Series("", index=frame.index)).astype(str).str.upper()
    frame = frame[~frame["display_ticker"].isin({t.upper() for t in exclude_tickers})].copy()

    score = pd.to_numeric(frame.get("light_anomaly_score"), errors="coerce")
    frame = frame[(score >= float(min_anomaly)) & (score <= float(max_anomaly))].copy()
    if frame.empty:
        return frame

    exchange_col = "light_exchange" if "light_exchange" in frame.columns else "source_exchange"
    market = str(market or "global").strip().lower()
    if market != "global" and exchange_col in frame.columns:
        groups = {
            "usa": {"US", "NYSE", "NASDAQ", "AMEX", "BATS"},
            "europe": {"LSE", "PA", "XETRA", "F", "MI", "SW", "AS", "BR", "MC", "LS", "ST", "CO", "HE", "OL", "WAR"},
            "asia": {"TSE", "HK", "SHE", "SZSE"},
            "canada": {"TO", "V"},
            "australia": {"AU", "AX"},
            "africa": {"JSE"},
        }
        allowed = groups.get(market)
        if allowed:
            frame = frame[frame[exchange_col].astype(str).str.upper().isin(allowed)].copy()

    if company_size != "all" and "light_market_cap_usd" in frame.columns:
        caps = pd.to_numeric(frame["light_market_cap_usd"], errors="coerce")
        if company_size == "large":
            frame = frame[caps >= 10_000_000_000].copy()
        elif company_size == "medium":
            frame = frame[(caps >= 2_000_000_000) & (caps < 10_000_000_000)].copy()
        elif company_size == "small":
            frame = frame[caps < 2_000_000_000].copy()

    if selected_sectors and "light_sector" in frame.columns:
        frame = frame[frame["light_sector"].astype(str).str.lower().isin(selected_sectors)].copy()

    if float(min_drawdown_pct) > 0 and "light_drawdown_250d_pct" in frame.columns:
        dd = pd.to_numeric(frame["light_drawdown_250d_pct"], errors="coerce")
        frame = frame[dd <= -abs(float(min_drawdown_pct))].copy()

    if int(min_average_volume) > 0 and "light_volume" in frame.columns:
        vol = pd.to_numeric(frame["light_volume"], errors="coerce")
        frame = frame[vol >= int(min_average_volume)].copy()

    if frame.empty:
        return frame

    frame["ticker"] = frame["display_ticker"]
    frame["last_close"] = pd.to_numeric(frame.get("light_last_price"), errors="coerce")
    frame["drawdown_52w_pct"] = pd.to_numeric(frame.get("light_drawdown_250d_pct"), errors="coerce")
    frame["anomaly_score"] = pd.to_numeric(frame.get("light_anomaly_score"), errors="coerce")
    frame["opportunity_score"] = pd.NA
    frame["value_trap_risk"] = pd.NA
    frame["valuation_score"] = pd.NA
    frame["financial_risk_score"] = pd.NA
    frame["distress_risk_score"] = pd.NA
    frame["dilution_risk_score"] = pd.NA
    frame["confidence_score"] = pd.NA
    frame["catalyst_label"] = "Non analizzato (Light)"
    frame["analysis_level"] = "light"
    frame["price_source"] = "eodhd_bulk_eod"
    frame["price_observed_at"] = frame.get("light_data_date")
    frame["price_is_delayed"] = True
    frame["price_validation"] = "ok"
    frame["exchange"] = frame.get("light_exchange", "")
    frame["sector"] = frame.get("light_sector", "")
    frame = frame.sort_values(
        ["anomaly_score", "light_market_cap_usd"],
        ascending=[False, False],
        na_position="last",
    )
    return frame

def get_dashboard(
    min_opportunity: float = 0,
    max_value_trap: float = 100,
    min_anomaly: float = 60,
    max_anomaly: float = 100,
    min_confidence: float = 0,
    top_n: int = 20,
    market: str = "global",
    company_size: str = "all",
    sectors: str = "",
    risk_profile: str = "balanced",
    min_valuation: float = 0,
    min_drawdown_pct: float = 0,
    min_average_volume: int = 0,
    event_filter: str = "all",
) -> dict:

    scan_time, market_mode, records = (
        load_latest_scan()
    )
    (
        light_scan_time,
        light_mode,
        universe_scanned,
        light_anomaly_candidates,
        light_records,
    ) = load_latest_light_scan()

    if not records and not light_records:
        return {
            "scan_time": None,
            "market_mode": market_mode,
            "top_anomalies": [],
            "stats": {
                "universe_scanned": int(universe_scanned or 0),
                "light_candidates": int(light_anomaly_candidates or 0),
                "deep_analyzed": 0,
                "analyzed": 0,
                "candidates": 0,
                "displayed": 0,
                "max_opportunity": None,
                "max_anomaly": None,
                "stale_prices": 0,
                "failed": 0,
            },
            "market_tension": get_market_tension(),
        }

    df = pd.DataFrame(records)

    if df.empty:
        valid = pd.DataFrame()
    elif "error" in df.columns:
        valid = df[
            df["error"].isna()
        ].copy()
    elif not df.empty:
        valid = df.copy()

    deep_analyzed_total = int(len(valid))
    deep_failed_total = int(len(df) - len(valid))

    market = str(market or "global").strip().lower()
    company_size = str(company_size or "all").strip().lower()
    risk_profile = str(risk_profile or "balanced").strip().lower()
    event_filter = str(event_filter or "all").strip().lower()

    if risk_profile == "conservative":
        max_value_trap = min(float(max_value_trap), 45.0)
        min_confidence = max(float(min_confidence), 45.0)
    elif risk_profile == "aggressive":
        max_value_trap = min(float(max_value_trap), 80.0)

    exchange_column = (
        "light_exchange" if "light_exchange" in valid.columns else "exchange"
    )
    if market != "global" and exchange_column in valid.columns:
        groups = {
            "usa": {"US", "NYSE", "NASDAQ", "AMEX"},
            "europe": {
                "LSE", "PA", "XETRA", "MI", "SW", "AS", "BR", "MC",
                "LS", "ST", "CO", "HE", "OL",
            },
            "asia": {"TSE", "HK", "SHE", "SZSE"},
            "canada": {"TO", "V"},
            "australia": {"AU", "AX"},
            "africa": {"JSE"},
        }
        allowed = groups.get(market)
        if allowed:
            valid = valid[
                valid[exchange_column].astype(str).str.upper().isin(allowed)
            ].copy()

    cap_columns = [
        column for column in ("market_cap_usd", "light_market_cap_usd")
        if column in valid.columns
    ]
    if company_size != "all":
        caps = pd.Series(float("nan"), index=valid.index, dtype="float64")
        for column in cap_columns:
            caps = caps.fillna(pd.to_numeric(valid[column], errors="coerce"))

        # Compatibilità con snapshot precedenti: il valore locale è utilizzato
        # soltanto quando la valuta è esplicitamente USD.
        currencies = valid.get(
            "currency", pd.Series("", index=valid.index, dtype=str)
        ).astype(str).str.upper()
        for column in ("market_cap", "light_market_cap"):
            if column in valid.columns:
                local_usd = pd.to_numeric(valid[column], errors="coerce").where(
                    currencies == "USD"
                )
                caps = caps.fillna(local_usd)

        if company_size == "large":
            valid = valid[caps >= 10_000_000_000].copy()
        elif company_size == "medium":
            valid = valid[(caps >= 2_000_000_000) & (caps < 10_000_000_000)].copy()
        elif company_size == "small":
            valid = valid[caps < 2_000_000_000].copy()

    selected_sectors = {
        item.strip().lower() for item in str(sectors or "").split(",") if item.strip()
    }
    sector_column = "light_sector" if "light_sector" in valid.columns else "sector"
    if selected_sectors and sector_column in valid.columns:
        valid = valid[
            valid[sector_column].astype(str).str.lower().isin(selected_sectors)
        ].copy()

    if float(min_drawdown_pct) > 0:
        drawdowns = pd.to_numeric(
            valid.get(
                "drawdown_52w_pct",
                pd.Series(float("nan"), index=valid.index),
            ),
            errors="coerce",
        )
        valid = valid[
            drawdowns <= -abs(float(min_drawdown_pct))
        ].copy()

    if int(min_average_volume) > 0:
        volumes = pd.to_numeric(
            valid.get(
                "light_volume",
                pd.Series(float("nan"), index=valid.index),
            ),
            errors="coerce",
        )
        valid = valid[
            volumes >= int(min_average_volume)
        ].copy()

    if event_filter != "all":
        catalyst_labels = valid.get(
            "catalyst_label", pd.Series("", index=valid.index, dtype=str)
        ).astype(str)
        if event_filter == "identified":
            valid = valid[
                ~catalyst_labels.isin({
                    "", "Non analizzato", "Nessun catalizzatore disponibile",
                })
            ].copy()
        elif event_filter == "earnings":
            earnings = valid.get(
                "earnings_related", pd.Series(False, index=valid.index)
            ).fillna(False).astype(bool)
            valid = valid[earnings].copy()
        elif event_filter == "structural":
            valid = valid[
                catalyst_labels == "Possibile rischio strutturale"
            ].copy()

    for numeric_column in (
        "opportunity_score", "value_trap_risk", "anomaly_score", "confidence_score"
    ):
        if numeric_column not in valid.columns:
            valid[numeric_column] = pd.NA
        valid[numeric_column] = pd.to_numeric(valid[numeric_column], errors="coerce")

    trap_ok = valid["value_trap_risk"].isna() | (
        valid["value_trap_risk"] <= max_value_trap
    )

    valuation_ok = (
        pd.Series(True, index=valid.index)
        if float(min_valuation) <= 0
        else (
            valid["valuation_score"].notna()
            & (valid["valuation_score"] >= float(min_valuation))
        )
    )

    candidates = valid[
        (
            valid["opportunity_score"]
            >= min_opportunity
        )
        & valuation_ok
        & trap_ok
        & (
            valid["anomaly_score"]
            >= min_anomaly
        )
        & (
            valid["anomaly_score"]
            <= max_anomaly
        )
        & (
            valid["confidence_score"]
            >= min_confidence
        )
    ].copy()

    deep_candidates = candidates.sort_values(
        ["anomaly_score", "opportunity_score"],
        ascending=[False, False],
        na_position="last",
    ).copy()
    if not deep_candidates.empty:
        deep_candidates["analysis_level"] = "deep"

    # Light-only rows remain browseable in simple anomaly bands. This is what
    # lets a 20k+ global scan surface moderate/normal anomalies even though only
    # the strongest ~300 receive the expensive Deep analysis automatically.
    light_allowed = (
        float(min_opportunity) <= 0
        and float(min_confidence) <= 0
        and float(min_valuation) <= 0
        and event_filter == "all"
    )
    light_candidates_frame = pd.DataFrame()
    if light_allowed:
        analyzed_tickers = set(
            valid.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper()
        )
        light_candidates_frame = _prepare_light_dashboard_frame(
            light_records,
            market=market,
            company_size=company_size,
            selected_sectors=selected_sectors,
            min_anomaly=min_anomaly,
            max_anomaly=max_anomaly,
            min_drawdown_pct=min_drawdown_pct,
            min_average_volume=min_average_volume,
            exclude_tickers=analyzed_tickers,
        )

    candidate_total = int(len(deep_candidates) + len(light_candidates_frame))
    display_frames = [frame for frame in (deep_candidates, light_candidates_frame) if not frame.empty]
    if display_frames:
        candidates = pd.concat(display_frames, ignore_index=True, sort=False)
        candidates["_deep_first"] = candidates.get(
            "analysis_level", pd.Series("deep", index=candidates.index)
        ).astype(str).eq("deep").astype(int)
        candidates = candidates.sort_values(
            ["anomaly_score", "_deep_first", "opportunity_score"],
            ascending=[False, False, False],
            na_position="last",
        ).head(int(top_n)).drop(columns=["_deep_first"], errors="ignore")
    else:
        candidates = pd.DataFrame()

    top = []
    quote_provider = None
    if not candidates.empty:
        try:
            quote_provider = _market_provider()
            if hasattr(quote_provider, "batch_latest_quotes"):
                quote_provider.batch_latest_quotes(
                    [
                        str(value)
                        for value in candidates.get(
                            "provider_ticker",
                            candidates.get("ticker", pd.Series(dtype=str)),
                        ).dropna()
                    ]
                )
        except Exception:
            quote_provider = None

    for _, r in candidates.iterrows():
        d = _row_to_dict(r)
        if quote_provider is not None:
            d = _refresh_record_quote(d, provider=quote_provider)

        analysis_level = str(d.get("analysis_level") or "deep").lower()
        if analysis_level == "light":
            classification_label = _light_classification(d.get("anomaly_score"))
            summary = _light_summary(d)
            data_gaps = ["Analisi Deep non ancora eseguita"]
        else:
            cls = build_ticker_narrative(d)
            classification_label = cls["classification"]["label"]
            summary = cls.get("summary")
            data_gaps = cls.get("data_gaps", [])

        price_meta = _price_metadata(d)
        display_ticker = str(d.get("ticker") or "").upper()

        top.append({
            "ticker": display_ticker,
            "provider_ticker": d.get("provider_ticker") or display_ticker,
            "company": d.get("company"),
            "price": d.get("last_close"),
            "drawdown_52w_pct": d.get("drawdown_52w_pct"),
            "anomaly_score": d.get("anomaly_score"),
            "opportunity_score": d.get("opportunity_score"),
            "value_trap_risk": d.get("value_trap_risk"),
            "valuation_score": d.get("valuation_score"),
            "financial_risk_score": d.get("financial_risk_score"),
            "distress_risk_score": d.get("distress_risk_score"),
            "dilution_risk_score": d.get("dilution_risk_score"),
            "confidence_score": d.get("confidence_score"),
            "catalyst_label": d.get("catalyst_label"),
            "classification": classification_label,
            "summary": summary,
            "data_gaps": data_gaps,
            "currency": d.get("currency") or "USD",
            "exchange": d.get("light_exchange") or d.get("exchange") or "",
            "sector": d.get("light_sector") or d.get("sector") or "",
            "analysis_level": analysis_level,
            **price_meta,
            "in_watchlist": is_in_watchlist(display_ticker),
        })

    return {
        "scan_time": scan_time or light_scan_time,
        "market_mode": market_mode or light_mode,
        "top_anomalies": top,
        "stats": {
            "universe_scanned": int(
                universe_scanned
                or (valid.iloc[0].get("global_universe_scanned", 0) if len(valid) else 0)
            ),
            "light_candidates": int(
                light_anomaly_candidates
                or (valid.iloc[0].get("light_anomaly_candidates", 0) if len(valid) else 0)
            ),
            "deep_analyzed": deep_analyzed_total,
            "analyzed": deep_analyzed_total,
            "candidates": candidate_total,
            "displayed": int(len(candidates)),
            "max_opportunity": (
                _finite_max(valid, "opportunity_score")
            ),
            "max_anomaly": (
                max(
                    [value for value in (
                        _finite_max(valid, "anomaly_score"),
                        _finite_max(light_candidates_frame, "anomaly_score") if not light_candidates_frame.empty else None,
                    ) if value is not None],
                    default=None,
                )
            ),
            "stale_prices": int(sum(
                1 for record in records
                if _price_metadata(record)["price_status"] in {"stale", "conflict", "unknown"}
            )),
            "failed": deep_failed_total,
        },
        "model_version": CONFIG.model_version,
        "minimum_quality_gate": {
            "min_confidence": min_confidence,
            "min_anomaly": min_anomaly,
            "max_anomaly": max_anomaly,
        },
        "active_filters": {
            "market": market,
            "company_size": company_size,
            "sectors": sorted(selected_sectors),
            "risk_profile": risk_profile,
            "min_anomaly": float(min_anomaly),
            "max_anomaly": float(max_anomaly),
            "max_value_trap": max_value_trap,
            "min_valuation": float(min_valuation),
            "min_drawdown_pct": float(min_drawdown_pct),
            "min_average_volume": int(min_average_volume),
            "event_filter": event_filter,
        },
    }


def get_ticker_detail(
    ticker: str,
) -> dict | None:

    _, _, records = (
        load_latest_scan()
    )

    if not records:
        return None

    requested = str(ticker).strip().upper()
    display = requested.split(".")[0]

    provider_matches = [
        r for r in records
        if str(r.get("provider_ticker") or "").upper() == requested
    ]
    display_matches = [
        r for r in records
        if str(r.get("ticker") or "").upper() == display
    ]
    match = provider_matches or display_matches

    if not match:
        return None

    if not provider_matches and len(match) > 1:
        match.sort(
            key=lambda record: (
                0 if bool(record.get("is_primary_listing", True)) else 1,
                str(record.get("provider_ticker") or ""),
            )
        )

    # On a single US detail page use the real-time EODHD WebSocket feed when
    # available. Global/home lists deliberately keep the cheaper delayed REST
    # snapshot so opening the dashboard never spawns hundreds of sockets.
    row = _refresh_record_quote(match[0], prefer_realtime=True)

    narrative = build_ticker_narrative(
        row
    )

    return {
        **row,
        **_price_metadata(row),
        "narrative": narrative,
        "in_watchlist": (
            is_in_watchlist(
                display
            )
        ),
    }


def add_watchlist_item(
    ticker: str,
) -> dict:

    detail = get_ticker_detail(
        ticker
    )

    if detail is None:
        return {
            "ok": False,
            "message": (
                f"Nessun dato recente per "
                f"{ticker.upper()}. "
                "Esegui prima una scansione."
            ),
        }

    ok = add_to_watchlist(
        ticker=ticker,
        company=detail.get(
            "company",
            "",
        ),
        price=detail.get(
            "last_close"
        ),
        anomaly_score=detail.get(
            "anomaly_score"
        ),
        opportunity_score=detail.get(
            "opportunity_score"
        ),
        catalyst_label=detail.get(
            "catalyst_label",
            "",
        ),
    )

    return {
        "ok": ok,
        "message": (
            "Aggiunto alla watchlist."
            if ok
            else (
                "Il titolo è già "
                "in watchlist."
            )
        ),
    }


def remove_watchlist_item(
    ticker: str,
) -> dict:

    remove_from_watchlist(
        ticker
    )

    return {
        "ok": True
    }


def get_watchlist() -> list[dict]:
    wl = list_watchlist()

    if wl.empty:
        return []

    _, _, records = (
        load_latest_scan()
    )

    latest_by_ticker = {
        str(
            r.get(
                "ticker",
                "",
            )
        ).upper(): r
        for r in records
    }

    out = []
    quote_provider = None
    try:
        quote_provider = _market_provider()
    except Exception:
        quote_provider = None

    for _, w in wl.iterrows():
        ticker = w["ticker"]

        latest = latest_by_ticker.get(
            ticker,
            {},
        )
        if latest:
            latest = _refresh_record_quote(latest, provider=quote_provider)

        current_price = latest.get(
            "last_close"
        )

        added_price = w.get(
            "price_at_add"
        )

        perf = None

        if (
            current_price is not None
            and added_price
        ):
            try:
                perf = round(
                    (
                        float(current_price)
                        / float(added_price)
                        - 1
                    )
                    * 100,
                    2,
                )
            except Exception:
                perf = None

        out.append({
            "ticker": ticker,
            "company": w.get(
                "company"
            ),
            "added_at": w.get(
                "added_at"
            ),
            "price_at_add": added_price,
            "current_price": current_price,
            "performance_pct": perf,
            "anomaly_score_at_add": (
                w.get(
                    "anomaly_score_at_add"
                )
            ),
            "anomaly_score_now": (
                latest.get(
                    "anomaly_score"
                )
            ),
            "opportunity_score_at_add": (
                w.get(
                    "opportunity_score_at_add"
                )
            ),
            "opportunity_score_now": (
                latest.get(
                    "opportunity_score"
                )
            ),
            "catalyst_label_at_add": (
                w.get("catalyst_label_at_add") or ""
            ),
            "catalyst_label_now": (
                latest.get("catalyst_label") or ""
            ),
            "has_new_event": bool(
                latest.get("catalyst_label")
                and latest.get("catalyst_label") != "Non analizzato"
                and latest.get("catalyst_label")
                != (w.get("catalyst_label_at_add") or "")
            ),
            "currency": latest.get("currency") or "USD",
            **(_price_metadata(latest) if latest else {}),
        })

    return out


def get_history(
    limit: int = 500,
) -> list[dict]:

    snapshots = load_snapshot_history(limit=limit)

    if snapshots:
        output = []
        for snapshot in snapshots:
            payload = snapshot.get("payload") or {}
            output.append({
                "snapshot_id": snapshot.get("snapshot_id"),
                "signal_time": snapshot.get("snapshot_time"),
                "ticker": snapshot.get("ticker"),
                "company": snapshot.get("company"),
                "price": snapshot.get("price"),
                "currency": snapshot.get("currency") or "USD",
                "price_source": snapshot.get("price_source"),
                "price_observed_at": snapshot.get("price_observed_at"),
                "anomaly_score": _clean_json_value(
                    payload.get("anomaly_score")
                ),
                "opportunity_score": _clean_json_value(
                    payload.get("opportunity_score")
                ),
                "recovery_potential": _clean_json_value(
                    payload.get("recovery_potential")
                ),
                "value_trap_risk": _clean_json_value(
                    payload.get("value_trap_risk")
                ),
                "catalyst_risk": _clean_json_value(
                    payload.get("catalyst_risk")
                ),
                "quality_score": _clean_json_value(
                    payload.get("quality_score")
                ),
                "confidence_score": _clean_json_value(
                    payload.get("confidence_score")
                ),
                "catalyst_label": payload.get("catalyst_label"),
                "model_version": snapshot.get("model_version"),
                "outcomes": snapshot.get("outcomes") or [],
            })
        return output

    hist = load_signals(
        limit=limit
    )

    if hist.empty:
        return []

    output = []
    for _, row in hist.iterrows():
        payload = {}
        try:
            payload = json.loads(row.get("payload_json") or "{}")
        except Exception:
            payload = {}
        output.append({
            "signal_time": row.get("signal_time"),
            "ticker": row.get("ticker"),
            "company": row.get("company"),
            "price": _clean_json_value(row.get("price")),
            "currency": payload.get("currency") or "USD",
            "price_source": payload.get("price_source"),
            "price_observed_at": payload.get("price_observed_at"),
            "anomaly_score": _clean_json_value(row.get("anomaly_score")),
            "opportunity_score": _clean_json_value(row.get("opportunity_score")),
            "recovery_potential": _clean_json_value(row.get("recovery_potential")),
            "value_trap_risk": _clean_json_value(row.get("value_trap_risk")),
            "catalyst_risk": _clean_json_value(row.get("catalyst_risk")),
            "quality_score": _clean_json_value(row.get("quality_score")),
            "model_version": payload.get("model_version") or CONFIG.model_version,
            "outcomes": [],
        })
    return output


def _provider_ticker_for(ticker: str) -> str:
    normalized = str(ticker).strip().upper()
    # A full provider symbol is already an unambiguous instrument identifier.
    # Do not collapse e.g. ABC.AU and ABC.US into the same display ticker.
    if "." in normalized:
        return normalized

    _, _, records = load_latest_scan()

    matches = [
        record for record in records
        if str(record.get("ticker", "")).upper() == normalized
    ]
    if matches:
        # Prefer a primary listing when metadata is available; otherwise keep
        # the exact provider symbol stored by the scanner.
        matches.sort(
            key=lambda record: (
                0 if bool(record.get("is_primary_listing", True)) else 1,
                str(record.get("provider_ticker") or ""),
            )
        )
        return str(matches[0].get("provider_ticker") or f"{normalized}.US").upper()

    return f"{normalized}.US"


def get_price_history(
    ticker: str,
    period: str = "1M",
) -> dict:
    """Price chart with true calendar windows and robust intraday fallback."""
    normalized_period = str(period).strip().upper()
    settings = {
        "1G": {"intraday": True, "days": 1, "interval": "5m", "daily_size": 8},
        "5G": {"intraday": True, "days": 7, "interval": "1h", "daily_size": 12},
        "1M": {"intraday": False, "outputsize": 45, "months": 1},
        "6M": {"intraday": False, "outputsize": 165, "months": 6},
        "1A": {"intraday": False, "outputsize": 285, "years": 1},
        "5A": {"intraday": False, "outputsize": 1400, "years": 5},
    }
    if normalized_period not in settings:
        raise ValueError("Periodo non valido. Usa 1G, 5G, 1M, 6M, 1A oppure 5A.")

    provider = _market_provider()
    provider_ticker = _provider_ticker_for(ticker)
    cfg = settings[normalized_period]
    note = None
    series_type = "provider_adjusted_history"

    if cfg.get("intraday") and hasattr(provider, "intraday_history"):
        try:
            frame = provider.intraday_history(
                provider_ticker,
                days=int(cfg["days"]),
                interval=str(cfg["interval"]),
            )
            series_type = f"provider_intraday_{cfg['interval']}"
        except Exception as error:
            frame = provider.daily_history(
                provider_ticker,
                outputsize=int(cfg["daily_size"]),
            ).tail(int(cfg["daily_size"]))
            series_type = "daily_fallback"
            note = (
                "Storico intraday temporaneamente non disponibile; "
                f"mostro chiusure giornaliere. Dettaglio provider: {error}"
            )
    else:
        frame = provider.daily_history(
            provider_ticker,
            outputsize=int(cfg["outputsize"]),
        )
        frame = frame.dropna(subset=["datetime", "close"]).copy()
        if not frame.empty:
            anchor = pd.Timestamp(frame["datetime"].max())
            if cfg.get("months"):
                cutoff = anchor - pd.DateOffset(months=int(cfg["months"]))
            else:
                cutoff = anchor - pd.DateOffset(years=int(cfg.get("years", 1)))
            trimmed = frame[pd.to_datetime(frame["datetime"]) >= cutoff].copy()
            if len(trimmed) >= 2:
                frame = trimmed

    if frame is None or frame.empty:
        raise RuntimeError(f"Storico prezzi non disponibile per {ticker.upper()}.")

    frame = frame.dropna(subset=["datetime", "close"]).sort_values("datetime").copy()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = 0.0 if column == "volume" else frame["close"]
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].fillna(0.0)
    for column in ("open", "high", "low"):
        frame[column] = frame[column].fillna(frame["close"])

    full_closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if full_closes.empty:
        raise RuntimeError(f"Prezzi non validi per {ticker.upper()}.")

    first_close = float(full_closes.iloc[0])
    last_close = float(full_closes.iloc[-1])
    change_pct = (last_close / first_close - 1) * 100 if first_close else None
    running_max = full_closes.cummax()
    drawdowns = (full_closes / running_max - 1) * 100
    period_high = float(full_closes.max())
    period_low = float(full_closes.min())
    max_drawdown = float(drawdowns.min())

    # Downsample only after statistics are computed, so peaks/troughs are never
    # lost from min/max/drawdown as happened in v2.2.
    display_frame = frame
    if len(display_frame) > 500:
        indexes = list(range(0, len(display_frame), max(1, math.ceil(len(display_frame) / 500))))
        if indexes[-1] != len(display_frame) - 1:
            indexes.append(len(display_frame) - 1)
        display_frame = display_frame.iloc[indexes].copy()

    points = []
    for _, row in display_frame.iterrows():
        timestamp = pd.Timestamp(row["datetime"])
        def number(value):
            try:
                value = float(value)
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                return None
        points.append({
            "time": timestamp.isoformat(),
            "open": number(row.get("open")),
            "high": number(row.get("high")),
            "low": number(row.get("low")),
            "close": number(row.get("close")),
            "volume": number(row.get("volume")) or 0.0,
        })

    current_quote = None
    if provider_ticker.upper().endswith(".US") and hasattr(provider, "realtime_trade_quote"):
        try:
            current_quote = provider.realtime_trade_quote(provider_ticker)
        except Exception as realtime_error:
            if note is None:
                note = f"Tempo reale USA temporaneamente non disponibile: {realtime_error}"
    if current_quote is None and hasattr(provider, "latest_quote"):
        try:
            current_quote = provider.latest_quote(provider_ticker)
        except Exception as error:
            if note is None:
                note = f"Quota recente non disponibile: {error}"

    raw_reference = last_close
    if "raw_close" in frame.columns:
        raw_values = pd.to_numeric(frame["raw_close"], errors="coerce").dropna()
        if not raw_values.empty:
            raw_reference = float(raw_values.iloc[-1])

    if current_quote and not quote_matches_reference(current_quote, raw_reference):
        current_quote = None
        consistency_note = (
            "Quota recente non coerente con la chiusura storica: "
            "nel riepilogo è stata mantenuta la chiusura verificata."
        )
        note = f"{note} {consistency_note}".strip() if note else consistency_note

    _, _, latest_records = load_latest_scan()
    latest_record = next(
        (
            item for item in latest_records
            if str(item.get("ticker", "")).upper() == str(ticker).strip().upper().split(".")[0]
        ),
        {},
    )

    current_source = current_quote.get("source") if current_quote else "historical_close"
    current_delayed = bool(current_quote.get("is_delayed", True)) if current_quote else True
    return {
        "ticker": str(ticker).strip().upper().split(".")[0],
        "provider_ticker": provider_ticker,
        "period": normalized_period,
        "series_type": series_type,
        "currency": latest_record.get("currency") or "USD",
        "points": points,
        "summary": {
            "first_close": first_close,
            "last_close": last_close,
            "change_pct": round(change_pct, 3) if change_pct is not None else None,
            "period_high": period_high,
            "period_low": period_low,
            "max_drawdown_pct": round(max_drawdown, 3),
            "current_price": (
                float(current_quote["price"])
                if current_quote and current_quote.get("price") is not None
                else last_close
            ),
            "current_price_observed_at": (
                current_quote.get("observed_at") if current_quote else points[-1]["time"]
            ),
            "current_price_source": current_source,
            "current_price_is_delayed": current_delayed,
            "current_price_status": (
                "market_closed"
                if current_quote and current_quote.get("source") == "eodhd_websocket_realtime"
                and str(current_quote.get("market_status") or "").lower() == "closed"
                else "extended_hours"
                if current_quote and current_quote.get("source") == "eodhd_websocket_realtime"
                and str(current_quote.get("market_status") or "").lower() in {"extended-hours", "extended_hours", "extended"}
                else "delayed" if current_quote and current_delayed
                else "last_close" if not current_quote
                else "live"
            ),
        },
        "note": note,
    }


def _local_search_rows() -> list[dict]:
    rows = []
    seen = set()

    for item in FALLBACK_LIVE_UNIVERSE:
        ticker = str(item["display_ticker"]).upper()
        seen.add(ticker)
        rows.append({
            "ticker": ticker,
            "provider_ticker": str(item["ticker"]).upper(),
            "company": str(item["company"]),
            "exchange": "US",
            "venue": "US",
            "currency": "USD",
            "country": "United States",
            "type": "Common Stock",
        })

    universe_path = Path(__file__).resolve().parent / "universe.csv"

    if universe_path.exists():
        try:
            universe = pd.read_csv(universe_path)

            for _, item in universe.iterrows():
                ticker = str(item.get("ticker", "")).upper()

                if not ticker or ticker in seen:
                    continue

                seen.add(ticker)
                rows.append({
                    "ticker": ticker,
                    "provider_ticker": f"{ticker}.US",
                    "company": str(item.get("company", ticker)),
                    "exchange": "US",
                    "type": "Common Stock",
                })
        except Exception:
            pass

    return rows


def search_tickers(
    query: str,
    limit: int = 12,
) -> list[dict]:
    text = str(query).strip()

    if len(text) < 1:
        return []

    lowered = text.lower()
    local = [
        item
        for item in _local_search_rows()
        if lowered in item["ticker"].lower()
        or lowered in item["company"].lower()
    ]

    results = list(local)

    if CONFIG.data_mode == "live":
        try:
            provider = _market_provider()

            if hasattr(provider, "search_symbols"):
                # For ticker-like queries ask the provider for the full search
                # window before ranking. A primary listing must not disappear
                # just because several foreign listings share the same code.
                provider_limit = 25 if text.replace(".", "").replace("-", "").isalnum() else limit
                results.extend(
                    provider.search_symbols(text, limit=provider_limit)
                )
        except Exception:
            pass

    exact = text.upper()
    is_symbol_like = text.replace(".", "").replace("-", "").isalnum()

    unique = []
    seen = set()
    for item in results:
        item = dict(item)
        key = str(item.get("provider_ticker", "")).upper()
        if not key or key in seen:
            continue
        seen.add(key)
        item.setdefault("venue", item.get("exchange") or "")
        item.setdefault("currency", "")
        item.setdefault("country", "")
        unique.append(item)

    # An exact ticker may exist on several exchanges.  Prefer the primary US
    # listing when it exists (NVDA.US before secondary NVDA listings), while
    # still keeping the other listings visible and labelled.
    def search_rank(item: dict):
        ticker_code = str(item.get("ticker") or "").upper()
        provider_symbol = str(item.get("provider_ticker") or "").upper()
        exchange = str(item.get("exchange") or "").upper()
        venue = str(item.get("venue") or "").upper()
        exact_code = ticker_code == exact or provider_symbol == exact
        provider_primary = bool(item.get("is_primary"))
        primary_us = provider_symbol.endswith(".US") or exchange in {
            "US", "NASDAQ", "NYSE", "AMEX", "BATS"
        } or venue in {"NASDAQ", "NYSE", "AMEX", "BATS"}
        return (
            0 if exact_code else 1,
            0 if exact_code and provider_primary else 1,
            0 if exact_code and primary_us else 1,
            0 if ticker_code.startswith(exact) else 1,
            str(item.get("company") or "").lower(),
        )

    unique.sort(key=search_rank)

    if is_symbol_like and not unique:
        unique.append({
            "ticker": exact.split(".")[0],
            "provider_ticker": exact if "." in exact else f"{exact}.US",
            "company": exact,
            "exchange": exact.rsplit(".", 1)[1] if "." in exact else "US",
            "venue": exact.rsplit(".", 1)[1] if "." in exact else "US",
            "currency": "USD" if "." not in exact or exact.endswith(".US") else "",
            "country": "",
            "type": "Ticker inserito manualmente",
        })

    return unique[: max(1, min(int(limit), 25))]


def analyze_ticker(
    ticker: str,
    provider_ticker: str | None = None,
    company: str | None = None,
    exchange: str | None = None,
    currency: str | None = None,
) -> dict:
    display_ticker = str(ticker).strip().upper().split(".")[0]

    if not display_ticker:
        raise ValueError("Ticker mancante.")

    api_ticker = str(
        provider_ticker
        or (ticker if "." in str(ticker) else f"{display_ticker}.US")
    ).strip().upper()

    provider = _market_provider()
    exchange = str(
        exchange or (api_ticker.rsplit(".", 1)[1] if "." in api_ticker else "US")
    ).strip().upper()
    benchmark = _market_benchmark(exchange)
    universe = pd.DataFrame([{
        "ticker": api_ticker,
        "display_ticker": display_ticker,
        "company": str(company or display_ticker),
        "sector_etf": benchmark,
        "benchmark_ticker": benchmark,
        "light_exchange": exchange,
        "currency": _currency_for(exchange, currency),
    }])

    result = scan_universe(
        universe,
        provider,
        include_sec=(
            _include_sec()
            and api_ticker.endswith(".US")
        ),
        catalyst_top_n=1,
    )

    if result.empty:
        raise RuntimeError(
            f"Nessun dato disponibile per {display_ticker}."
        )

    row = _row_to_dict(result.iloc[0])
    row["model_version"] = CONFIG.model_version
    row["data_provider"] = CONFIG.market_data_provider

    if row.get("error"):
        raise RuntimeError(str(row["error"]))

    _, market_mode, records = load_latest_scan()
    merged = [
        record
        for record in records
        if str(record.get("provider_ticker") or "").upper() != api_ticker
    ]
    merged.append(row)
    merged_frame = pd.DataFrame(merged)

    save_latest_scan(
        merged_frame,
        market_mode=market_mode or CONFIG.data_mode,
    )
    save_signals(pd.DataFrame([row]))
    save_signal_snapshots(
        pd.DataFrame([row]),
        model_version=CONFIG.model_version,
    )

    detail = get_ticker_detail(api_ticker)

    if detail is None:
        raise RuntimeError(
            f"Analisi non disponibile per {display_ticker}."
        )

    return detail


def _history_close_as_of(frame: pd.DataFrame, target) -> float | None:
    if frame is None or frame.empty:
        return None
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
    target_time = pd.Timestamp(target)
    if target_time.tzinfo is None:
        target_time = target_time.tz_localize("UTC")
    eligible = data[data["datetime"] <= target_time]
    if not eligible.empty:
        value = pd.to_numeric(eligible.iloc[-1]["close"], errors="coerce")
        return float(value) if pd.notna(value) else None

    # Se lo storico comincia dopo il target usiamo la prima seduta successiva,
    # non l'ultima dell'intera serie (che introdurrebbe look-ahead bias).
    future = data[data["datetime"] >= target_time]
    if future.empty:
        return None
    value = pd.to_numeric(future.iloc[0]["close"], errors="coerce")
    return float(value) if pd.notna(value) else None


def update_due_outcomes(limit: int = 100) -> dict:
    """Aggiorna gli esiti maturati; non modifica mai lo snapshot originale."""
    provider = _market_provider()
    due = list_due_outcomes(limit=limit)
    completed = 0
    failed = 0

    for item in due:
        try:
            payload = {}
            try:
                import json
                payload = json.loads(item.get("payload_json") or "{}")
            except Exception:
                payload = {}

            ticker_frame = provider.daily_history(
                item["provider_ticker"], outputsize=400
            )
            benchmark = item.get("benchmark_ticker") or "SPY.US"
            benchmark_frame = provider.daily_history(benchmark, outputsize=400)

            signal_time = pd.Timestamp(item["snapshot_time"])
            due_time = pd.Timestamp(item["due_at"])
            now = pd.Timestamp.now(tz="UTC")
            if due_time.tzinfo is None:
                due_time = due_time.tz_localize("UTC")

            # Rileggiamo anche il prezzo storico della data del segnale dalla
            # serie oggi rettificata. In questo modo split e altre corporate
            # action successive non producono vittorie o perdite artificiali.
            adjusted_signal_price = _history_close_as_of(
                ticker_frame,
                signal_time,
            )
            signal_price = (
                adjusted_signal_price
                or payload.get("calculation_close")
                or item.get("signal_price")
            )
            outcome_price = _history_close_as_of(ticker_frame, min(due_time, now))
            benchmark_signal = _history_close_as_of(benchmark_frame, signal_time)
            benchmark_outcome = _history_close_as_of(benchmark_frame, min(due_time, now))

            if not signal_price or outcome_price is None:
                raise RuntimeError("Prezzo sufficiente per l'outcome non disponibile.")

            absolute_return = (float(outcome_price) / float(signal_price) - 1) * 100
            benchmark_return = None
            if benchmark_signal and benchmark_outcome is not None:
                benchmark_return = (
                    float(benchmark_outcome) / float(benchmark_signal) - 1
                ) * 100

            data = ticker_frame.copy()
            data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
            window = data[
                (data["datetime"] >= signal_time)
                & (data["datetime"] <= min(due_time, now))
            ]
            closes = pd.to_numeric(window["close"], errors="coerce").dropna()
            max_drawdown = (
                float((closes / float(signal_price) - 1).min() * 100)
                if not closes.empty else None
            )
            recovered_positions = [
                index for index, value in enumerate(closes.tolist(), start=1)
                if value >= float(signal_price)
            ]

            save_outcome(item["outcome_id"], {
                "status": "complete",
                "outcome_price": outcome_price,
                "benchmark_signal_price": benchmark_signal,
                "benchmark_outcome_price": benchmark_outcome,
                "absolute_return_pct": absolute_return,
                "benchmark_return_pct": benchmark_return,
                "relative_return_pct": (
                    absolute_return - benchmark_return
                    if benchmark_return is not None else None
                ),
                "max_drawdown_pct": max_drawdown,
                "recovered": bool(recovered_positions),
                "recovery_sessions": (
                    recovered_positions[0] if recovered_positions else None
                ),
            })
            completed += 1
        except Exception as error:
            save_outcome(item["outcome_id"], {
                "status": "pending",
                "error_message": str(error),
            })
            failed += 1

    return {
        "due": len(due),
        "completed": completed,
        "failed": failed,
        "learning": learning_summary(),
    }


def get_learning_summary() -> dict:
    return learning_summary()


def submit_feedback(
    ticker: str,
    feedback_type: str,
    note: str = "",
) -> dict:
    feedback_id = save_user_feedback(
        ticker=ticker,
        feedback_type=feedback_type,
        note=note,
    )
    return {
        "ok": True,
        "feedback_id": feedback_id,
        "message": "Feedback salvato per la valutazione controllata del modello.",
    }


def get_diagnostics() -> dict:
    scan_time, market_mode, records = load_latest_scan()
    stale = [
        record for record in records
        if _price_metadata(record)["price_status"] in {"stale", "conflict", "unknown"}
    ]
    return {
        "app_version": CONFIG.app_version,
        "model_version": CONFIG.model_version,
        "data_mode": CONFIG.data_mode,
        "provider": CONFIG.market_data_provider,
        "configured_exchanges": [
            value.strip() for value in CONFIG.screener_exchanges.split(",")
            if value.strip()
        ],
        "target_light_universe": CONFIG.light_universe_limit,
        "deep_candidate_limit": CONFIG.deep_candidate_limit,
        "daily_cache_ttl_minutes": CONFIG.daily_cache_ttl_minutes,
        "last_scan_time": scan_time,
        "last_scan_mode": market_mode,
        "last_scan_records": len(records),
        "stale_or_conflicting_prices": len(stale),
        "storage": storage_diagnostics(),
        "learning": learning_summary(),
        "api_key_configured": bool(CONFIG.api_key),
        "sec_user_agent_configured": bool(
            CONFIG.sec_user_agent
            and "your-email@example.com" not in CONFIG.sec_user_agent.lower()
        ),
        "secrets_exposed": False,
    }
