"""
Market Anomaly API — backend REST per l'app Android.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CONFIG
import service


app = FastAPI(
    title="Market Anomaly API",
    version="1.3",
    description=(
        "Backend per l'app Market Anomaly: "
        "scanner di anomalie post-earnings."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(x_api_key: str | None):
    if CONFIG.api_key and x_api_key != CONFIG.api_key:
        raise HTTPException(
            status_code=401,
            detail="API key mancante o non valida.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": CONFIG.app_name,
        "data_mode": CONFIG.data_mode,
    }


@app.post("/api/scan")
def trigger_scan(
    limit: int = Query(40, ge=10, le=2000),
    catalyst_top_n: int = Query(5, ge=1, le=25),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    return service.start_scan_background(
        limit=min(limit, 40),
        catalyst_top_n=catalyst_top_n,
    )


@app.get("/api/scan/status")
def scan_status(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_scan_status()


@app.get("/api/dashboard")
def dashboard(
    min_opportunity: float = Query(55, ge=0, le=100),
    max_value_trap: float = Query(65, ge=0, le=100),
    top_n: int = Query(20, ge=1, le=100),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    return service.get_dashboard(
        min_opportunity,
        max_value_trap,
        top_n,
    )


@app.get("/api/ticker/{ticker}")
def ticker_detail(
    ticker: str,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    detail = service.get_ticker_detail(ticker)

    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Nessun dato recente per {ticker.upper()}.",
        )

    return detail


class WatchlistAddRequest(BaseModel):
    ticker: str


@app.get("/api/watchlist")
def watchlist(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_watchlist()


@app.post("/api/watchlist")
def watchlist_add(
    body: WatchlistAddRequest,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.add_watchlist_item(body.ticker)


@app.delete("/api/watchlist/{ticker}")
def watchlist_remove(
    ticker: str,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.remove_watchlist_item(ticker)


@app.get("/api/history")
def history(
    limit: int = Query(500, ge=1, le=5000),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_history(limit=limit)
