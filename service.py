"""
Layer di servizio: incapsula il motore esistente (scanner, model, catalyst_engine,
storage) in funzioni pulite, richiamabili da un'API stateless.

Questo modulo NON reimplementa la logica quantitativa: riusa scanner.py così
com'è. Sostituisce solo st.session_state con storage.py (SQLite) come fonte
di verità condivisa tra richieste/utenti.
"""
from __future__ import annotations
import math
import pandas as pd

from config import CONFIG
from scanner import scan_universe
from providers.demo import DemoProvider
from providers.twelve_data import TwelveDataProvider
from providers.pit_demo import DemoPointInTimeFundamentals
from providers.sec_edgar import SecEdgarProvider
from universe_manager import normalize_universe, build_demo_historical_universe, active_snapshot
from storage import (
    save_signals, load_signals, save_latest_scan, load_latest_scan,
    add_to_watchlist, remove_from_watchlist, list_watchlist, is_in_watchlist,
)
from narrative import build_ticker_narrative


def _market_provider():
    if CONFIG.data_mode == "live":
        if not CONFIG.twelve_data_api_key:
            raise RuntimeError(
                "MARKET_ANOMALY_DATA_MODE=live ma TWELVE_DATA_API_KEY non è impostata sul server."
            )
        return TwelveDataProvider(CONFIG.twelve_data_api_key, cache_dir=CONFIG.price_cache_dir)
    return DemoProvider()


def _include_sec() -> bool:
    return CONFIG.data_mode == "live"


def _clean_json_value(v):
    """SQLite/JSON non gestiscono NaN: normalizza in None."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _row_to_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    return {k: _clean_json_value(v) for k, v in d.items()}


def run_scan(limit: int = 100, catalyst_top_n: int = 7) -> dict:
    """Esegue una scansione completa e la salva come 'ultima scansione' + storico."""
    universe = normalize_universe(build_demo_historical_universe(250))
    live_u = active_snapshot(universe, pd.Timestamp.today()).head(int(limit))

    df = scan_universe(
        live_u,
        _market_provider(),
        include_sec=_include_sec(),
        catalyst_top_n=int(catalyst_top_n),
    )

    save_latest_scan(df, market_mode=CONFIG.data_mode)

    valid = df[df["error"].isna()].copy() if "error" in df.columns else df
    if not valid.empty:
        save_signals(valid)

    return {
        "scanned": int(len(df)),
        "valid": int(len(valid)),
        "data_mode": CONFIG.data_mode,
    }


def get_dashboard(min_opportunity: float = 55, max_value_trap: float = 65, top_n: int = 20) -> dict:
    scan_time, market_mode, records = load_latest_scan()
    if not records:
        return {
            "scan_time": None,
            "market_mode": market_mode,
            "top_anomalies": [],
            "stats": {"analyzed": 0, "candidates": 0, "max_opportunity": None, "max_anomaly": None},
        }

    df = pd.DataFrame(records)
    valid = df[df["error"].isna()] if "error" in df.columns else df
    candidates = valid[
        (valid["opportunity_score"] >= min_opportunity) & (valid["value_trap_risk"] <= max_value_trap)
    ].copy()
    candidates = candidates.sort_values("opportunity_score", ascending=False).head(int(top_n))

    top = []
    for _, r in candidates.iterrows():
        d = _row_to_dict(r)
        cls = build_ticker_narrative(d)["classification"]
        top.append({
            "ticker": d.get("ticker"),
            "company": d.get("company"),
            "price": d.get("last_close"),
            "drawdown_52w_pct": d.get("drawdown_52w_pct"),
            "anomaly_score": d.get("anomaly_score"),
            "opportunity_score": d.get("opportunity_score"),
            "value_trap_risk": d.get("value_trap_risk"),
            "catalyst_label": d.get("catalyst_label"),
            "classification": cls["label"],
            "in_watchlist": is_in_watchlist(d.get("ticker", "")),
        })

    return {
        "scan_time": scan_time,
        "market_mode": market_mode,
        "top_anomalies": top,
        "stats": {
            "analyzed": int(len(valid)),
            "candidates": int(len(candidates)),
            "max_opportunity": float(valid["opportunity_score"].max()) if len(valid) else None,
            "max_anomaly": float(valid["anomaly_score"].max()) if len(valid) else None,
        },
    }


def get_ticker_detail(ticker: str) -> dict | None:
    _, _, records = load_latest_scan()
    if not records:
        return None
    ticker = ticker.upper()
    match = [r for r in records if str(r.get("ticker", "")).upper() == ticker]
    if not match:
        return None
    row = match[0]
    narrative = build_ticker_narrative(row)
    return {
        **row,
        "narrative": narrative,
        "in_watchlist": is_in_watchlist(ticker),
    }


def add_watchlist_item(ticker: str) -> dict:
    detail = get_ticker_detail(ticker)
    if detail is None:
        return {"ok": False, "message": f"Nessun dato recente per {ticker.upper()}. Esegui prima una scansione."}
    ok = add_to_watchlist(
        ticker=ticker,
        company=detail.get("company", ""),
        price=detail.get("last_close"),
        anomaly_score=detail.get("anomaly_score"),
        opportunity_score=detail.get("opportunity_score"),
    )
    return {"ok": ok, "message": "Aggiunto alla watchlist." if ok else "Il titolo è già in watchlist."}


def remove_watchlist_item(ticker: str) -> dict:
    remove_from_watchlist(ticker)
    return {"ok": True}


def get_watchlist() -> list[dict]:
    wl = list_watchlist()
    if wl.empty:
        return []
    _, _, records = load_latest_scan()
    latest_by_ticker = {str(r.get("ticker", "")).upper(): r for r in records}

    out = []
    for _, w in wl.iterrows():
        ticker = w["ticker"]
        latest = latest_by_ticker.get(ticker, {})
        current_price = latest.get("last_close")
        added_price = w.get("price_at_add")
        perf = None
        if current_price is not None and added_price:
            try:
                perf = round((float(current_price) / float(added_price) - 1) * 100, 2)
            except Exception:
                perf = None
        out.append({
            "ticker": ticker,
            "company": w.get("company"),
            "added_at": w.get("added_at"),
            "price_at_add": added_price,
            "current_price": current_price,
            "performance_pct": perf,
            "anomaly_score_at_add": w.get("anomaly_score_at_add"),
            "anomaly_score_now": latest.get("anomaly_score"),
            "opportunity_score_at_add": w.get("opportunity_score_at_add"),
            "opportunity_score_now": latest.get("opportunity_score"),
        })
    return out


def get_history(limit: int = 500) -> list[dict]:
    hist = load_signals(limit=limit)
    if hist.empty:
        return []
    cols = [
        "signal_time", "ticker", "company", "price", "anomaly_score",
        "opportunity_score", "recovery_potential", "value_trap_risk",
        "catalyst_risk", "quality_score",
    ]
    cols = [c for c in cols if c in hist.columns]
    return hist[cols].to_dict(orient="records")
