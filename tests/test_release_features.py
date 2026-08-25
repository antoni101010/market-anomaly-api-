from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import service
from api.main import app


class _PriceProvider:
    def daily_history(self, symbol: str, outputsize: int = 300):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        closes = [100.0, 105.0, 102.0, 110.0, 108.0, 120.0]

        return pd.DataFrame({
            "datetime": dates,
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        })


def test_price_history_summary(monkeypatch):
    monkeypatch.setattr(service, "_market_provider", lambda: _PriceProvider())
    monkeypatch.setattr(service, "_provider_ticker_for", lambda _: "META.US")

    result = service.get_price_history("meta", period="1M")

    assert result["ticker"] == "META"
    assert result["provider_ticker"] == "META.US"
    assert len(result["points"]) == 6
    assert result["summary"]["change_pct"] == 20.0
    assert result["summary"]["period_low"] == 100.0
    assert result["summary"]["period_high"] == 120.0
    assert result["summary"]["max_drawdown_pct"] < 0


def test_price_history_rejects_unknown_period():
    with pytest.raises(ValueError, match="Periodo non valido"):
        service.get_price_history("META", period="2A")


def test_local_ticker_search_is_deduplicated(monkeypatch):
    monkeypatch.setattr(
        service,
        "_local_search_rows",
        lambda: [
            {
                "ticker": "META",
                "provider_ticker": "META.US",
                "company": "Meta Platforms",
                "exchange": "US",
                "type": "Common Stock",
            },
            {
                "ticker": "META",
                "provider_ticker": "META.US",
                "company": "Meta Platforms",
                "exchange": "US",
                "type": "Common Stock",
            },
        ],
    )

    results = service.search_tickers("meta")

    assert len(results) == 1
    assert results[0]["ticker"] == "META"


def test_release_routes(monkeypatch):
    monkeypatch.setattr(
        service,
        "get_price_history",
        lambda ticker, period: {
            "ticker": ticker.upper(),
            "provider_ticker": f"{ticker.upper()}.US",
            "period": period,
            "points": [],
            "summary": {},
            "note": None,
        },
    )
    monkeypatch.setattr(
        service,
        "search_tickers",
        lambda query, limit=12: [{
            "ticker": query.upper(),
            "provider_ticker": f"{query.upper()}.US",
            "company": query.upper(),
            "exchange": "US",
            "type": "Common Stock",
        }][:limit],
    )

    client = TestClient(app)
    chart = client.get("/api/ticker/META/prices", params={"period": "6M"})
    search = client.get("/api/search", params={"q": "meta"})

    assert chart.status_code == 200
    assert chart.json()["period"] == "6M"
    assert search.status_code == 200
    assert search.json()[0]["ticker"] == "META"
