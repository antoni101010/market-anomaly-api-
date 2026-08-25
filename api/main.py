"""
Market Anomaly API — backend REST per l'app Android.
"""
from __future__ import annotations
import secrets
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CONFIG
import service


app = FastAPI(
    title="Market Anomaly API",
    version=CONFIG.app_version,
    description=(
        "Backend per l'app Market Anomaly: "
        "ricerca statistica su anomalie, fondamentali, eventi e tensione globale dei mercati."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(x_api_key: str | None):
    if CONFIG.api_key and (
        not x_api_key
        or not secrets.compare_digest(x_api_key, CONFIG.api_key)
    ):
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
        "provider": CONFIG.market_data_provider,
        "version": CONFIG.app_version,
        "model_version": CONFIG.model_version,
        "real_data_only": True,
        "api_protected": bool(CONFIG.api_key),
    }


@app.get("/api/legal/current")
def legal_current():
    return service.get_legal_current()


class LegalAcceptanceRequest(BaseModel):
    installation_id: str = Field(..., min_length=8, max_length=160)
    terms_version: str = Field(..., min_length=1, max_length=64)
    privacy_version: str = Field(..., min_length=1, max_length=64)
    app_version: str = Field(default="", max_length=64)
    platform: str = Field(default="", max_length=64)
    terms_accepted: bool = True
    privacy_notice_acknowledged: bool = True


@app.post("/api/legal/acceptance")
def legal_acceptance(
    body: LegalAcceptanceRequest,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    try:
        return service.record_legal_acceptance(
            body.installation_id,
            terms_version=body.terms_version,
            privacy_version=body.privacy_version,
            app_version=body.app_version,
            platform=body.platform,
            terms_accepted=body.terms_accepted,
            privacy_notice_acknowledged=body.privacy_notice_acknowledged,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.delete("/api/legal/installation/{installation_id}")
def legal_delete_installation(
    installation_id: str,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.delete_installation_data(installation_id)


@app.get("/api/market-tension")
def market_tension(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_market_tension()


@app.post("/api/market-tension/refresh")
def market_tension_refresh(
    force: bool = Query(False),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.start_market_tension_background(force=force)


@app.get("/api/market-tension/history")
def market_tension_history(
    limit: int = Query(90, ge=1, le=3650),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_market_tension_history(limit=limit)


@app.post("/api/scan")
def trigger_scan(
    limit: int = Query(100, ge=10, le=500),
    catalyst_top_n: int = Query(5, ge=1, le=25),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    return service.start_scan_background(
        limit=min(limit, CONFIG.deep_candidate_limit),
        catalyst_top_n=catalyst_top_n,
    )


@app.get("/api/scan/status")
def scan_status(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_scan_status()


class HistoricalBackfillRequest(BaseModel):
    years: int | None = Field(default=None, ge=1, le=50)
    as_of: str | None = Field(default=None, max_length=32)
    tickers: list[str] = Field(default_factory=list, max_length=10000)
    limit: int | None = Field(default=None, ge=1, le=10000)
    resume: bool = True


@app.post("/api/historical-backfill")
def trigger_historical_backfill(
    body: HistoricalBackfillRequest,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    try:
        return service.start_historical_backfill_background(
            years=body.years,
            as_of=body.as_of,
            tickers=body.tickers,
            limit=body.limit,
            resume=body.resume,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/historical-backfill/status")
def historical_backfill_status(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_historical_backfill_status()


@app.get("/api/historical-learning/stats")
def historical_learning_stats(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    try:
        return service.get_historical_learning_stats()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/dashboard")
def dashboard(
    min_opportunity: float = Query(45, ge=0, le=100),
    max_value_trap: float = Query(65, ge=0, le=100),
    min_anomaly: float = Query(20, ge=0, le=100),
    min_confidence: float = Query(25, ge=0, le=100),
    top_n: int = Query(20, ge=1, le=100),
    market: str = Query("global", max_length=24),
    company_size: str = Query("all", max_length=24),
    sectors: str = Query("", max_length=500),
    risk_profile: str = Query("balanced", max_length=24),
    min_valuation: float = Query(0, ge=0, le=100),
    min_drawdown_pct: float = Query(0, ge=0, le=100),
    min_average_volume: int = Query(0, ge=0, le=100_000_000),
    event_filter: str = Query("all", max_length=32),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    return service.get_dashboard(
        min_opportunity=min_opportunity,
        max_value_trap=max_value_trap,
        min_anomaly=min_anomaly,
        min_confidence=min_confidence,
        top_n=top_n,
        market=market,
        company_size=company_size,
        sectors=sectors,
        risk_profile=risk_profile,
        min_valuation=min_valuation,
        min_drawdown_pct=min_drawdown_pct,
        min_average_volume=min_average_volume,
        event_filter=event_filter,
    )


@app.get("/api/ticker/{ticker}/prices")
def ticker_prices(
    ticker: str,
    period: str = Query("1M"),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    try:
        return service.get_price_history(
            ticker=ticker,
            period=period,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error


@app.get("/api/search")
def ticker_search(
    q: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(12, ge=1, le=25),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.search_tickers(q, limit=limit)


class AnalyzeTickerRequest(BaseModel):
    ticker: str
    provider_ticker: str | None = None
    company: str | None = None


@app.post("/api/analyze")
def analyze_ticker(
    body: AnalyzeTickerRequest,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    try:
        return service.analyze_ticker(
            ticker=body.ticker,
            provider_ticker=body.provider_ticker,
            company=body.company,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error


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


class FeedbackRequest(BaseModel):
    ticker: str
    feedback_type: str
    note: str = ""


@app.post("/api/feedback")
def feedback(
    body: FeedbackRequest,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    try:
        return service.submit_feedback(
            ticker=body.ticker,
            feedback_type=body.feedback_type,
            note=body.note,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/learning")
def learning(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_learning_summary()


@app.post("/api/outcomes/update")
def update_outcomes(
    limit: int = Query(100, ge=1, le=500),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    try:
        return service.update_due_outcomes(limit=limit)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/diagnostics")
def diagnostics(
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    return service.get_diagnostics()
