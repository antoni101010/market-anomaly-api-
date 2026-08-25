from __future__ import annotations

import math

import numpy as np
import pandas as pd

import service
from fundamentals import completeness_details
from providers.eodhd import EODHDProvider
from scanner import overall_confidence


def test_bulk_global_light_scanner_surfaces_app_like_anomaly(monkeypatch, tmp_path):
    provider = EODHDProvider(api_key="test", cache_dir=str(tmp_path))
    bulk = pd.DataFrame([
        {
            "code": "APP",
            "type": "Common Stock",
            "adjusted_close": 298.59,
            "MarketCapitalization": 100_000_000_000,
            "avgvol_50d": 4_000_000,
            "avgvol_14d": 3_000_000,
            "volume": 10_000_000,
            "change_p": -6.0,
            "hi_250d": 745.0,
            "lo_250d": 219.0,
            "ema_50d": 430.0,
            "ema_200d": 480.0,
            "name": "AppLovin Corp",
        },
        {
            "code": "CALM",
            "type": "Common Stock",
            "adjusted_close": 100.0,
            "MarketCapitalization": 30_000_000_000,
            "avgvol_50d": 1_000_000,
            "avgvol_14d": 1_000_000,
            "volume": 1_000_000,
            "change_p": -0.5,
            "hi_250d": 110.0,
            "lo_250d": 80.0,
            "ema_50d": 101.0,
            "ema_200d": 98.0,
            "name": "Calm Corp",
        },
    ])
    monkeypatch.setattr(provider, "bulk_eod_extended", lambda exchange: bulk.copy())

    universe = provider.bulk_market_universe(
        ["NASDAQ"],
        min_avg_volume=50_000,
        min_price=1.0,
        min_market_cap_usd=500_000_000,
    )

    assert len(universe) == 2
    assert universe.iloc[0]["display_ticker"] == "APP"
    assert universe.iloc[0]["light_anomaly_score"] > 50
    assert universe.iloc[0]["light_drawdown_250d_pct"] < -50

    selected = service._select_deep_candidates(universe, limit=2)
    assert selected.iloc[0]["display_ticker"] == "APP"


def test_global_exchange_core_survives_old_narrow_render_env(monkeypatch):
    class Config:
        screener_exchanges = "us,lse"

    monkeypatch.setattr(service, "CONFIG", Config())
    exchanges = service._global_scan_exchanges()
    assert "US" in exchanges
    assert "NASDAQ" not in exchanges
    assert "NYSE" not in exchanges
    assert "TSE" in exchanges
    assert "HK" in exchanges
    assert "AU" in exchanges
    assert "LSE" in exchanges


def test_positive_fcf_cash_runway_is_not_counted_missing():
    metrics = {
        "cash_runway_months": None,
        "cash_runway_status": "not_applicable_positive_fcf",
    }
    details = completeness_details(metrics)
    assert "cash_runway_months" not in details["missing_fields"]
    assert details["available_fields"] >= 1


def test_overall_confidence_cannot_be_100_without_catalyst_context():
    row = {
        "fundamental_confidence_score": 100.0,
        "price_validation": "provider_matches_eod",
        "price_observed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "catalyst_label": "Non analizzato",
        "data_completeness": {"groups": {"valuation": 100.0}},
    }
    score = overall_confidence(row)
    assert score < 85
    assert score > 0


class _LongHistoryProvider:
    def daily_history(self, symbol: str, outputsize: int = 300):
        periods = max(1500, outputsize)
        dates = pd.date_range("2021-01-01", periods=periods, freq="B")
        close = np.linspace(100.0, 200.0, periods)
        # A sharp peak that display downsampling is allowed to skip, but summary is not.
        close[733] = 1000.0
        return pd.DataFrame({
            "datetime": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "raw_close": close,
            "volume": np.ones(periods) * 1000,
        })

    def latest_quote(self, symbol: str):
        raise RuntimeError("no quote in synthetic test")


def test_price_windows_are_calendar_based_and_stats_use_full_series(monkeypatch):
    provider = _LongHistoryProvider()
    monkeypatch.setattr(service, "_market_provider", lambda: provider)
    monkeypatch.setattr(service, "_provider_ticker_for", lambda _: "APP.US")
    monkeypatch.setattr(service, "load_latest_scan", lambda: (None, "live", []))

    one_month = service.get_price_history("APP", period="1M")
    start = pd.Timestamp(one_month["points"][0]["time"])
    end = pd.Timestamp(one_month["points"][-1]["time"])
    assert 25 <= (end - start).days <= 35

    five_year = service.get_price_history("APP", period="5A")
    assert five_year["summary"]["period_high"] == 1000.0
    assert len(five_year["points"]) <= 501
    assert math.isfinite(five_year["summary"]["max_drawdown_pct"])


def test_deep_selection_uses_full_300_budget_on_20k_universe():
    size = 20_000
    frame = pd.DataFrame({
        "ticker": [f"T{i:05d}.US" for i in range(size)],
        "display_ticker": [f"T{i:05d}" for i in range(size)],
        "light_anomaly_score": np.linspace(100.0, 0.0, size),
        "light_market_cap_usd": np.linspace(10_000_000_000, 500_000_000, size),
        "source_exchange": ["NASDAQ", "NYSE", "LSE", "TSE", "AU"] * (size // 5),
    })

    selected = service._select_deep_candidates(frame, limit=300)

    assert len(selected) == 300
    assert selected["ticker"].is_unique
    assert selected.iloc[0]["ticker"] == "T00000.US"


def test_nvda_like_corrupt_price_dependent_fundamentals_are_reconciled():
    from scanner import reconcile_market_metrics

    metrics = {
        "fundamentals_currency": "USD",
        "primary_ticker": "NVDA.US",
        "shares_outstanding": 24_250_000_000,
        "eps_ttm": 6.53,
        # Values deliberately model the bad screen the user observed.
        "market_cap": 336_430_000_000_000,
        "pe_ratio": 1.4,
        "price_to_sales": None,
        "revenue_ttm": 253_490_000_000,
        "free_cash_flow_ttm": 120_000_000_000,
    }
    technical = {
        "raw_eod_close": 208.48,
        "last_close": 208.48,
    }

    fixed = reconcile_market_metrics(
        metrics,
        technical,
        provider_ticker="NVDA.US",
        listing_currency="USD",
        fx_rate=1.0,
    )

    expected_cap = 208.48 * 24_250_000_000
    assert fixed["is_primary_listing"] is True
    assert math.isclose(fixed["market_cap"], expected_cap, rel_tol=1e-9)
    assert fixed["market_cap"] < 6_000_000_000_000
    assert 30 < fixed["pe_ratio"] < 35
    assert 18 < fixed["price_to_sales"] < 22
    assert fixed["data_validation_status"] == "reconciled"
    assert len(fixed["data_validation_warnings"]) >= 2


def test_secondary_listing_withholds_ambiguous_price_dependent_fundamentals():
    from scanner import reconcile_market_metrics

    fixed = reconcile_market_metrics(
        {
            "fundamentals_currency": "ARS",
            "primary_ticker": "NVDA.US",
            "shares_outstanding": 24_250_000_000,
            "eps_ttm": 6.53,
            "market_cap": 33_643_000_000_000,
            "pe_ratio": 1.4,
            "price_to_sales": 0.2,
            "revenue_ttm": 253_490_000_000,
            "free_cash_flow_ttm": 120_000_000_000,
        },
        {"raw_eod_close": 14_040.0, "last_close": 14_040.0},
        provider_ticker="NVDA.BA",
        listing_currency="ARS",
        fx_rate=None,
    )

    assert fixed["is_primary_listing"] is False
    assert fixed["market_cap"] is None
    assert fixed["pe_ratio"] is None
    assert fixed["price_to_sales"] is None
    assert fixed["fcf_yield_pct"] is None
    assert fixed["data_validation_status"] == "secondary_listing"
    assert fixed["fundamental_consistency_score"] <= 35


def test_exact_search_prioritizes_primary_listing(monkeypatch):
    class Provider:
        def search_symbols(self, query, limit=12):
            assert limit == 25
            return [
                {
                    "ticker": "NVDA",
                    "provider_ticker": "NVDA.BA",
                    "company": "NVIDIA Corporation",
                    "exchange": "BA",
                    "venue": "Buenos Aires",
                    "currency": "ARS",
                    "country": "Argentina",
                    "is_primary": False,
                    "type": "Common Stock",
                },
                {
                    "ticker": "NVDA",
                    "provider_ticker": "NVDA.US",
                    "company": "NVIDIA Corporation",
                    "exchange": "US",
                    "venue": "NASDAQ",
                    "currency": "USD",
                    "country": "USA",
                    "is_primary": True,
                    "type": "Common Stock",
                },
            ]

    monkeypatch.setattr(service, "_market_provider", lambda: Provider())
    monkeypatch.setattr(service, "_local_search_rows", lambda: [])

    results = service.search_tickers("NVDA", limit=12)

    assert results[0]["provider_ticker"] == "NVDA.US"
    assert results[0]["is_primary"] is True


def test_intraday_null_fields_are_sanitized(monkeypatch, tmp_path):
    provider = EODHDProvider(api_key="test", cache_dir=str(tmp_path))
    payload = [
        {
            "datetime": "2026-08-25 13:30:00",
            "open": None,
            "high": None,
            "low": None,
            "close": "100.125",
            "volume": None,
        },
        {
            "datetime": "2026-08-25 13:35:00",
            "open": "100.125",
            "high": "101.0",
            "low": "99.75",
            "close": "100.5",
            "volume": "12345",
        },
    ]
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: payload)

    frame = provider.intraday_history("APP.US", days=1, interval="5m")

    assert len(frame) == 2
    assert frame["close"].notna().all()
    assert frame["open"].notna().all()
    assert frame["high"].notna().all()
    assert frame["low"].notna().all()
    assert frame["volume"].notna().all()
    assert frame.iloc[0]["open"] == frame.iloc[0]["close"]
    assert frame.iloc[0]["volume"] == 0.0


def test_dashboard_can_browse_normal_light_band_without_deep(monkeypatch):
    monkeypatch.setattr(service, "load_latest_scan", lambda: (None, "live", []))
    monkeypatch.setattr(
        service,
        "load_latest_light_scan",
        lambda: (
            "2026-08-25T12:00:00+00:00",
            "live",
            20_000,
            500,
            [
                {
                    "ticker": "NORMAL.US",
                    "display_ticker": "NORMAL",
                    "company": "Normal Anomaly Corp",
                    "light_anomaly_score": 30.0,
                    "light_last_price": 12.345,
                    "light_drawdown_250d_pct": -25.0,
                    "light_market_cap_usd": 2_000_000_000,
                    "light_volume": 1_000_000,
                    "light_exchange": "NASDAQ",
                    "currency": "USD",
                    "light_data_date": "2026-08-25",
                },
                {
                    "ticker": "STRONG.US",
                    "display_ticker": "STRONG",
                    "company": "Strong Anomaly Corp",
                    "light_anomaly_score": 72.0,
                    "light_last_price": 50.0,
                    "light_drawdown_250d_pct": -55.0,
                    "light_market_cap_usd": 5_000_000_000,
                    "light_volume": 2_000_000,
                    "light_exchange": "NYSE",
                    "currency": "USD",
                    "light_data_date": "2026-08-25",
                },
            ],
        ),
    )
    monkeypatch.setattr(service, "get_market_tension", lambda: {"status": "unavailable"})
    monkeypatch.setattr(service, "is_in_watchlist", lambda ticker: False)

    class Provider:
        def batch_latest_quotes(self, symbols):
            return {}

        def latest_quote(self, symbol):
            raise RuntimeError("synthetic no quote")

    monkeypatch.setattr(service, "_market_provider", lambda: Provider())

    result = service.get_dashboard(
        min_opportunity=0,
        min_confidence=0,
        min_valuation=0,
        min_anomaly=20,
        max_anomaly=39.999,
        top_n=30,
    )

    assert result["stats"]["universe_scanned"] == 20_000
    assert result["stats"]["deep_analyzed"] == 0
    assert result["stats"]["displayed"] == 1
    assert result["top_anomalies"][0]["ticker"] == "NORMAL"
    assert result["top_anomalies"][0]["provider_ticker"] == "NORMAL.US"
    assert result["top_anomalies"][0]["analysis_level"] == "light"
    assert result["top_anomalies"][0]["opportunity_score"] is None


def test_provider_ticker_prevents_same_symbol_cross_market_collision(monkeypatch):
    records = [
        {
            "ticker": "ABC",
            "provider_ticker": "ABC.US",
            "company": "ABC United States",
            "is_primary_listing": True,
            "last_close": 10.0,
        },
        {
            "ticker": "ABC",
            "provider_ticker": "ABC.AU",
            "company": "ABC Australia",
            "is_primary_listing": True,
            "last_close": 20.0,
        },
    ]
    monkeypatch.setattr(service, "load_latest_scan", lambda: (None, "live", records))
    monkeypatch.setattr(
        service,
        "_refresh_record_quote",
        lambda row, provider=None, prefer_realtime=False: dict(row),
    )
    monkeypatch.setattr(
        service,
        "build_ticker_narrative",
        lambda row: {"classification": {"label": "TEST"}, "summary": "", "data_gaps": []},
    )
    monkeypatch.setattr(service, "is_in_watchlist", lambda ticker: False)

    assert service._provider_ticker_for("ABC.AU") == "ABC.AU"
    detail = service.get_ticker_detail("ABC.AU")
    assert detail is not None
    assert detail["company"] == "ABC Australia"
    assert detail["provider_ticker"] == "ABC.AU"


def test_global_exchange_normalizes_stale_us_venue_env(monkeypatch):
    class Config:
        screener_exchanges = "nasdaq,nyse,amex,bats,lse"

    monkeypatch.setattr(service, "CONFIG", Config())
    exchanges = service._global_scan_exchanges()
    assert exchanges.count("US") == 1
    assert "NASDAQ" not in exchanges
    assert "NYSE" not in exchanges
    assert "AMEX" not in exchanges
    assert "BATS" not in exchanges


def test_price_history_prefers_realtime_us_quote(monkeypatch):
    class Provider:
        def daily_history(self, symbol, outputsize=300):
            dates = pd.date_range("2026-07-20", periods=30, freq="B")
            close = np.linspace(190.0, 208.0, len(dates))
            return pd.DataFrame({
                "datetime": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "raw_close": close,
                "volume": np.ones(len(dates)) * 1_000_000,
            })

        def realtime_trade_quote(self, symbol):
            assert symbol == "NVDA.US"
            return {
                "price": 209.25,
                "observed_at": "2026-08-25T14:00:00+00:00",
                "source": "eodhd_websocket_realtime",
                "is_delayed": False,
                "market_status": "open",
            }

        def latest_quote(self, symbol):
            raise AssertionError("Delayed REST quote should not be used when realtime succeeds")

    monkeypatch.setattr(service, "_market_provider", lambda: Provider())
    monkeypatch.setattr(service, "_provider_ticker_for", lambda _: "NVDA.US")
    monkeypatch.setattr(service, "load_latest_scan", lambda: (None, "live", []))

    history = service.get_price_history("NVDA.US", "1M")
    assert history["summary"]["current_price"] == 209.25
    assert history["summary"]["current_price_source"] == "eodhd_websocket_realtime"
    assert history["summary"]["current_price_status"] == "live"
    assert history["summary"]["current_price_is_delayed"] is False
