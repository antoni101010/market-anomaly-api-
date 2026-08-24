"""
Layer di servizio:
collega scanner, provider dati, storage e API.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone

import pandas as pd

from config import CONFIG
from scanner import (
    scan_universe,
    build_light_universe,
)
from providers.demo import DemoProvider
from providers.twelve_data import (
    TwelveDataProvider,
)
from providers.eodhd import EODHDProvider

from universe_manager import (
    normalize_universe,
    build_demo_historical_universe,
    active_snapshot,
)

from storage import (
    save_signals,
    load_signals,
    save_latest_scan,
    load_latest_scan,
    add_to_watchlist,
    remove_from_watchlist,
    list_watchlist,
    is_in_watchlist,
    save_scan_state,
    load_scan_state,
)

from narrative import (
    build_ticker_narrative,
)


def _market_provider():
    if CONFIG.data_mode == "live":

        if (
            CONFIG.market_data_provider
            == "eodhd"
        ):
            if not CONFIG.eodhd_api_key:
                raise RuntimeError(
                    "EODHD_API_KEY non è "
                    "impostata sul server."
                )

            return EODHDProvider(
                CONFIG.eodhd_api_key,
                cache_dir=(
                    CONFIG.price_cache_dir
                ),
            )

        if (
            CONFIG.market_data_provider
            == "twelve_data"
        ):
            if not CONFIG.twelve_data_api_key:
                raise RuntimeError(
                    "TWELVE_DATA_API_KEY non è "
                    "impostata sul server."
                )

            return TwelveDataProvider(
                CONFIG.twelve_data_api_key,
                cache_dir=(
                    CONFIG.price_cache_dir
                ),
            )

        raise RuntimeError(
            "Provider non riconosciuto. "
            "Usa eodhd oppure twelve_data."
        )

    return DemoProvider()


def _include_sec() -> bool:
    return (
        CONFIG.data_mode
        == "live"
    )


def _clean_json_value(value):
    if (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return None

    return value


def _row_to_dict(
    row: pd.Series,
) -> dict:

    data = row.to_dict()

    return {
        key: _clean_json_value(value)
        for key, value in data.items()
    }


def _build_live_shortlist(
    provider,
    limit: int,
) -> pd.DataFrame:
    """
    Prima fase veloce:

    usa lo screener EODHD e crea
    una shortlist reale.
    """

    shortlist = build_light_universe(
        market_provider=provider,
        exchanges=("us",),
        max_return_1d_pct=-8.0,
        min_avg_volume=200_000,
        min_price=2.0,
        min_market_cap=500_000_000,
        limit_per_exchange=max(
            20,
            min(
                int(limit),
                100,
            ),
        ),
    )

    if shortlist is None:
        raise RuntimeError(
            "Il provider selezionato "
            "non supporta il Light Scanner."
        )

    if shortlist.empty:
        return shortlist

    shortlist = shortlist.head(
        int(limit)
    )

    return shortlist


def run_scan(
    limit: int = 40,
    catalyst_top_n: int = 5,
) -> dict:

    provider = _market_provider()

    if CONFIG.data_mode == "live":
        live_universe = (
            _build_live_shortlist(
                provider,
                limit=limit,
            )
        )

        if live_universe.empty:
            empty = pd.DataFrame()

            save_latest_scan(
                empty,
                market_mode="live",
            )

            return {
                "scanned": 0,
                "valid": 0,
                "data_mode": "live",
                "provider": (
                    CONFIG.market_data_provider
                ),
                "light_candidates": 0,
            }

    else:
        universe = normalize_universe(
            build_demo_historical_universe(
                250
            )
        )

        live_universe = active_snapshot(
            universe,
            pd.Timestamp.today(),
        ).head(
            int(limit)
        )

    result = scan_universe(
        live_universe,
        provider,
        include_sec=_include_sec(),
        catalyst_top_n=int(
            catalyst_top_n
        ),
    )

    save_latest_scan(
        result,
        market_mode=CONFIG.data_mode,
    )

    valid = (
        result[
            result["error"].isna()
        ].copy()
        if "error" in result.columns
        else result
    )

    if not valid.empty:
        save_signals(
            valid
        )

    return {
        "scanned": int(
            len(result)
        ),
        "valid": int(
            len(valid)
        ),
        "data_mode": CONFIG.data_mode,
        "provider": (
            CONFIG.market_data_provider
            if CONFIG.data_mode == "live"
            else "demo"
        ),
        "light_candidates": int(
            len(live_universe)
        ),
    }


_scan_state = {
    "status": "idle",
    "message": "",
    "limit": 0,
    "started_at": None,
    "finished_at": None,
}

_scan_lock = threading.Lock()


_persisted_state = (
    load_scan_state()
)

if _persisted_state:
    _scan_state.update(
        _persisted_state
    )

    if (
        _scan_state["status"]
        == "running"
    ):
        _scan_state.update(
            status="error",
            message=(
                "La scansione è stata "
                "interrotta dal riavvio "
                "del server. Riprova."
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
            catalyst_top_n=(
                catalyst_top_n
            ),
        )

        _set_scan_state(
            status="done",
            message=(
                f"Completata: "
                f"{result['valid']} titoli "
                f"validi su "
                f"{result['scanned']} "
                f"analizzati."
            ),
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

    except Exception as error:
        _set_scan_state(
            status="error",
            message=str(error),
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )


def start_scan_background(
    limit: int = 40,
    catalyst_top_n: int = 5,
) -> dict:

    with _scan_lock:
        if (
            _scan_state["status"]
            == "running"
        ):
            return {
                "ok": False,
                "message": (
                    "Una scansione è "
                    "già in corso."
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
            "Scansione avviata "
            "in background."
        ),
    }


def get_dashboard(
    min_opportunity: float = 55,
    max_value_trap: float = 65,
    top_n: int = 20,
) -> dict:

    (
        scan_time,
        market_mode,
        records,
    ) = load_latest_scan()

    if not records:
        return {
            "scan_time": None,
            "market_mode": market_mode,
            "top_anomalies": [],
            "stats": {
                "analyzed": 0,
                "candidates": 0,
                "max_opportunity": None,
                "max_anomaly": None,
            },
        }

    data = pd.DataFrame(
        records
    )

    if "error" in data.columns:
        valid = data[
            data["error"].isna()
        ]

    else:
        valid = data

    candidates = valid[
        (
            valid["opportunity_score"]
            >= min_opportunity
        )
        & (
            valid["value_trap_risk"]
            <= max_value_trap
        )
    ].copy()

    candidates = (
        candidates.sort_values(
            "opportunity_score",
            ascending=False,
        )
        .head(
            int(top_n)
        )
    )

    top = []

    for _, row in candidates.iterrows():
        item = _row_to_dict(
            row
        )

        classification = (
            build_ticker_narrative(
                item
            )["classification"]
        )

        top.append({
            "ticker": item.get(
                "ticker"
            ),
            "company": item.get(
                "company"
            ),
            "price": item.get(
                "last_close"
            ),
            "drawdown_52w_pct": item.get(
                "drawdown_52w_pct"
            ),
            "anomaly_score": item.get(
                "anomaly_score"
            ),
            "opportunity_score": item.get(
                "opportunity_score"
            ),
            "value_trap_risk": item.get(
                "value_trap_risk"
            ),
            "valuation_score": item.get(
                "valuation_score"
            ),
            "financial_risk_score": item.get(
                "financial_risk_score"
            ),
            "distress_risk_score": item.get(
                "distress_risk_score"
            ),
            "dilution_risk_score": item.get(
                "dilution_risk_score"
            ),
            "confidence_score": item.get(
                "confidence_score"
            ),
            "catalyst_label": item.get(
                "catalyst_label"
            ),
            "classification": (
                classification["label"]
            ),
            "in_watchlist": is_in_watchlist(
                item.get(
                    "ticker",
                    "",
                )
            ),
        })

    return {
        "scan_time": scan_time,
        "market_mode": market_mode,
        "top_anomalies": top,
        "stats": {
            "analyzed": int(
                len(valid)
            ),
            "candidates": int(
                len(candidates)
            ),
            "max_opportunity": (
                float(
                    valid[
                        "opportunity_score"
                    ].max()
                )
                if len(valid)
                else None
            ),
            "max_anomaly": (
                float(
                    valid[
                        "anomaly_score"
                    ].max()
                )
                if len(valid)
                else None
            ),
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

    ticker = ticker.upper()

    matches = [
        record
        for record in records
        if str(
            record.get(
                "ticker",
                "",
            )
        ).upper()
        == ticker
    ]

    if not matches:
        return None

    row = matches[0]

    narrative = (
        build_ticker_narrative(
            row
        )
    )

    return {
        **row,
        "narrative": narrative,
        "in_watchlist": (
            is_in_watchlist(
                ticker
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
                "Nessun dato recente "
                f"per {ticker.upper()}. "
                "Esegui prima una scansione."
            ),
        }

    added = add_to_watchlist(
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
    )

    return {
        "ok": added,
        "message": (
            "Aggiunto alla watchlist."
            if added
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
    watchlist = list_watchlist()

    if watchlist.empty:
        return []

    _, _, records = (
        load_latest_scan()
    )

    latest_by_ticker = {
        str(
            record.get(
                "ticker",
                "",
            )
        ).upper(): record
        for record in records
    }

    output = []

    for _, watched in watchlist.iterrows():
        ticker = watched[
            "ticker"
        ]

        latest = latest_by_ticker.get(
            ticker,
            {},
        )

        current_price = latest.get(
            "last_close"
        )

        added_price = watched.get(
            "price_at_add"
        )

        performance = None

        if (
            current_price is not None
            and added_price
        ):
            try:
                performance = round(
                    (
                        float(current_price)
                        / float(added_price)
                        - 1
                    )
                    * 100,
                    2,
                )

            except Exception:
                performance = None

        output.append({
            "ticker": ticker,
            "company": watched.get(
                "company"
            ),
            "added_at": watched.get(
                "added_at"
            ),
            "price_at_add": (
                added_price
            ),
            "current_price": (
                current_price
            ),
            "performance_pct": (
                performance
            ),
            "anomaly_score_at_add": (
                watched.get(
                    "anomaly_score_at_add"
                )
            ),
            "anomaly_score_now": (
                latest.get(
                    "anomaly_score"
                )
            ),
            "opportunity_score_at_add": (
                watched.get(
                    "opportunity_score_at_add"
                )
            ),
            "opportunity_score_now": (
                latest.get(
                    "opportunity_score"
                )
            ),
        })

    return output


def get_history(
    limit: int = 500,
) -> list[dict]:

    history = load_signals(
        limit=limit
    )

    if history.empty:
        return []

    columns = [
        "signal_time",
        "ticker",
        "company",
        "price",
        "anomaly_score",
        "opportunity_score",
        "recovery_potential",
        "value_trap_risk",
        "catalyst_risk",
        "quality_score",
    ]

    columns = [
        column
        for column in columns
        if column in history.columns
    ]

    return history[
        columns
    ].to_dict(
        orient="records"
    )
