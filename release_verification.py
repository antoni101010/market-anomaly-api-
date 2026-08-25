"""Verifica offline ripetibile delle funzioni critiche della release 2.1."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # L'ambiente di verifica del pacchetto può non avere ancora installato
    # requests. Le prove sotto sono totalmente offline e usano solo parser e
    # provider finti, quindi basta uno stub d'importazione; in produzione il
    # pacchetto reale viene installato da requirements.txt.
    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")
        requests_stub.Session = object
        requests_stub.get = lambda *args, **kwargs: None
        adapters_stub = types.ModuleType("requests.adapters")
        adapters_stub.HTTPAdapter = object
        urllib3_stub = types.ModuleType("urllib3")
        urllib3_util_stub = types.ModuleType("urllib3.util")
        urllib3_retry_stub = types.ModuleType("urllib3.util.retry")
        urllib3_retry_stub.Retry = object
        sys.modules["requests"] = requests_stub
        sys.modules["requests.adapters"] = adapters_stub
        sys.modules["urllib3"] = urllib3_stub
        sys.modules["urllib3.util"] = urllib3_util_stub
        sys.modules["urllib3.util.retry"] = urllib3_retry_stub

    with tempfile.TemporaryDirectory(prefix="market_anomaly_verify_") as temp_dir:
        os.environ["MARKET_ANOMALY_DB"] = f"{temp_dir}/verify.db"
        os.environ["MARKET_ANOMALY_BACKUP_DIR"] = f"{temp_dir}/backups"

        # Simula la tabella watchlist della release precedente: l'import del
        # nuovo storage deve aggiungere la colonna eventi senza perdere righe.
        with sqlite3.connect(os.environ["MARKET_ANOMALY_DB"]) as connection:
            connection.execute(
                """
                CREATE TABLE watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    company TEXT,
                    added_at TEXT NOT NULL,
                    price_at_add REAL,
                    anomaly_score_at_add REAL,
                    opportunity_score_at_add REAL,
                    notes TEXT
                )
                """
            )

        from fundamentals import enrich_fundamental_scores, valuation_score
        from catalyst_engine import classify_catalysts
        from model import live_score
        from narrative import build_ticker_narrative
        from providers.eodhd import EODHDProvider
        from providers.sec_edgar import SecEdgarProvider
        from historical_learning import (
            HistoricalLearningConfig,
            mine_point_in_time_events,
            run_historical_backfill,
        )
        from scanner import quote_matches_reference
        import service
        import storage

        assert len(service._fallback_live_universe(200)) >= 25

        assert valuation_score({}) is None
        empty_scores = enrich_fundamental_scores({})
        assert empty_scores["valuation_score"] is None
        assert empty_scores["quality_score"] is None
        assert empty_scores["financial_risk_score"] is None
        assert classify_catalysts([], [])["catalyst_risk"] is None

        equal_components = {
            "score_drawdown": 40,
            "score_rsi": 40,
            "score_volume": 40,
            "score_momentum": 40,
            "score_shock": 40,
            "score_market_relative": 40,
            "score_sector_relative": 40,
        }
        # Una qualità assente viene esclusa e non trasformata in 0 o 50.
        assert live_score(equal_components, None) == 40.0

        first = build_ticker_narrative({
            "opportunity_score": 52,
            "confidence_score": 40,
            "drawdown_52w_pct": -31,
            "anomaly_score": 65,
            "valuation_score": 30,
            "pe_ratio": 44,
            "missing_fundamental_fields": ["current_ratio", "forward_pe"],
            "catalyst_label": "Nessun catalizzatore disponibile",
        })
        second = build_ticker_narrative({
            "opportunity_score": 68,
            "confidence_score": 82,
            "drawdown_52w_pct": -18,
            "anomaly_score": 72,
            "valuation_score": 76,
            "quality_score": 81,
            "catalyst_label": "Catalizzatore negativo potenzialmente temporaneo",
        })
        assert first["summary"] != second["summary"]
        assert "liquidità corrente" in " ".join(first["data_gaps"])
        assert first["classification"]["label"] != "MOVIMENTO NON PRIORITARIO"

        history = EODHDProvider._parse_history(
            [
                {
                    "date": "2026-01-02",
                    "open": 98,
                    "high": 102,
                    "low": 97,
                    "close": 100,
                    "adjusted_close": 50,
                    "volume": 1000,
                },
                {
                    "date": "2026-01-05",
                    "open": 51,
                    "high": 53,
                    "low": 50,
                    "close": 52,
                    "adjusted_close": 52,
                    "volume": 1200,
                },
            ],
            "TEST.US",
            20,
        )
        assert history.iloc[0]["raw_close"] == 100
        assert history.iloc[0]["close"] == 50
        assert history.iloc[0]["open"] == 49

        no_lookahead = pd.DataFrame({
            "datetime": pd.to_datetime(["2026-01-05", "2026-01-06"], utc=True),
            "close": [10.0, 99.0],
        })
        assert service._history_close_as_of(
            no_lookahead,
            "2026-01-01T00:00:00+00:00",
        ) == 10.0

        fx_provider = object.__new__(EODHDProvider)
        fx_provider.daily_history = lambda symbol, outputsize=5: pd.DataFrame({
            "close": [150.0 if symbol == "USDJPY.FOREX" else 1.1]
        })
        assert round(fx_provider.currency_to_usd_rate("JPY"), 6) == round(1 / 150, 6)
        assert fx_provider.currency_to_usd_rate("EUR") == 1.1
        parsed_quote = fx_provider._quote_from_payload(
            {"code": "SAP.XETRA", "close": 191.5, "timestamp": 1787600000},
            "SAP.XETRA",
        )
        assert parsed_quote["price"] == 191.5
        assert parsed_quote["provider_ticker"] == "SAP.XETRA"
        assert quote_matches_reference(
            {"price": 105, "previous_close": 100},
            100,
        )
        assert not quote_matches_reference(
            {"price": 1050, "previous_close": 1000},
            100,
        )
        missing_time_quote = fx_provider._quote_from_payload(
            {"code": "SAP.XETRA", "close": 191.5},
            "SAP.XETRA",
        )
        assert missing_time_quote["observed_at"] is None

        sec_provider = SecEdgarProvider(cache_dir=temp_dir)
        sec_provider.ticker_to_cik = lambda: {"TEST": "0000000001"}
        sec_calls = []

        class SecResponse:
            def json(self):
                return {
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "filingDate": ["2026-01-02"],
                            "accessionNumber": ["0000000001-26-000001"],
                            "primaryDocument": ["test.htm"],
                        }
                    }
                }

        def fake_sec_get(url):
            sec_calls.append(url)
            return SecResponse()

        sec_provider._get = fake_sec_get
        assert len(sec_provider.recent_filings("TEST")) == 1
        assert len(sec_provider.recent_filings("TEST")) == 1
        assert len(sec_calls) == 1

        snapshot_time = datetime.now(timezone.utc).isoformat()
        frame = pd.DataFrame([{
            "ticker": "TEST",
            "provider_ticker": "TEST.US",
            "company": "Test Company",
            "last_close": 100.0,
            "price_observed_at": snapshot_time,
            "price_source": "verification",
            "benchmark_ticker": "SPY.US",
            "currency": "USD",
            "confidence_score": 80.0,
            "anomaly_score": 70.0,
            "valuation_score": 60.0,
            "opportunity_score": 65.0,
            "value_trap_risk": 25.0,
            "error": None,
        }])
        assert storage.save_signal_snapshots(frame, snapshot_time=snapshot_time) == 1
        assert storage.learning_summary()["snapshots"] == 1
        assert storage.learning_summary()["outcomes_pending"] == 6
        snapshot_history = storage.load_snapshot_history()
        assert len(snapshot_history) == 1
        assert [
            item["horizon_sessions"]
            for item in snapshot_history[0]["outcomes"]
        ] == [1, 3, 7, 30, 90, 180]

        with storage._connect() as connection:
            first_outcome_id = connection.execute(
                """
                SELECT id FROM signal_outcomes
                WHERE snapshot_id=? AND horizon_sessions=1
                """,
                (snapshot_history[0]["snapshot_id"],),
            ).fetchone()[0]
        storage.save_outcome(first_outcome_id, {
            "status": "complete",
            "outcome_price": 105,
            "absolute_return_pct": 5,
            "relative_return_pct": 3,
            "max_drawdown_pct": -2,
            "recovered": True,
            "recovery_sessions": 1,
        })
        api_history = service.get_history()
        assert api_history[0]["outcomes"][0]["status"] == "complete"
        assert api_history[0]["outcomes"][0]["absolute_return_pct"] == 5

        assert storage.add_to_watchlist(
            "TEST",
            "Test Company",
            100,
            70,
            65,
            catalyst_label="Evento iniziale",
        )
        refreshed_frame = frame.copy()
        refreshed_frame["catalyst_label"] = "Evento aggiornato"
        storage.save_latest_scan(refreshed_frame, "live")
        watchlist = service.get_watchlist()
        assert watchlist[0]["catalyst_label_at_add"] == "Evento iniziale"
        assert watchlist[0]["catalyst_label_now"] == "Evento aggiornato"
        assert watchlist[0]["has_new_event"] is True
        feedback_id = storage.save_user_feedback("TEST", "useful")
        assert feedback_id > 0
        with storage._connect() as connection:
            linked_snapshot = connection.execute(
                "SELECT snapshot_id FROM user_feedback WHERE id=?",
                (feedback_id,),
            ).fetchone()
        assert linked_snapshot and linked_snapshot[0] is not None
        assert storage.create_database_backup() is not None
        assert storage.diagnostics()["backup_count"] == 1

        dashboard_frame = pd.DataFrame([
            {
                "ticker": "SAP", "company": "SAP SE", "last_close": 190,
                "currency": "EUR", "light_exchange": "XETRA",
                "light_sector": "Technology", "light_market_cap": 220_000_000_000,
                "light_market_cap_usd": 250_000_000_000,
                "light_volume": 900_000,
                "opportunity_score": 60, "anomaly_score": 55,
                "confidence_score": 70, "value_trap_risk": None,
                "valuation_score": 65, "drawdown_52w_pct": -20,
                "price_observed_at": snapshot_time, "price_source": "verification",
                "price_validation": "ok", "catalyst_label": "Causa non classificata",
                "error": None,
            },
            {
                "ticker": "TEST", "company": "Test Inc", "last_close": 100,
                "currency": "USD", "light_exchange": "US",
                "light_sector": "Technology", "light_market_cap": 5_000_000_000,
                "light_market_cap_usd": 5_000_000_000,
                "light_volume": 300_000,
                "opportunity_score": 58, "anomaly_score": 50,
                "confidence_score": 65, "value_trap_risk": 20,
                "valuation_score": 60, "drawdown_52w_pct": -18,
                "price_observed_at": snapshot_time, "price_source": "verification",
                "price_validation": "ok", "catalyst_label": "Causa non classificata",
                "error": None,
            },
        ])
        storage.save_latest_scan(dashboard_frame, "live")
        europe = service.get_dashboard(
            min_opportunity=0,
            max_value_trap=65,
            min_anomaly=0,
            min_confidence=0,
            top_n=20,
            market="europe",
            company_size="large",
        )
        assert [item["ticker"] for item in europe["top_anomalies"]] == ["SAP"]
        assert europe["top_anomalies"][0]["currency"] == "EUR"

        advanced = service.get_dashboard(
            min_opportunity=0,
            max_value_trap=65,
            min_anomaly=0,
            min_confidence=0,
            top_n=20,
            min_valuation=62,
            min_drawdown_pct=19,
            min_average_volume=500_000,
            event_filter="identified",
        )
        assert [item["ticker"] for item in advanced["top_anomalies"]] == ["SAP"]
        assert advanced["active_filters"]["min_valuation"] == 62

        class Provider:
            def daily_history(self, symbol, outputsize=300):
                dates = pd.date_range("2026-01-01", periods=8, freq="D")
                closes = [100, 102, 101, 104, 106, 108, 107, 110]
                return pd.DataFrame({
                    "datetime": dates,
                    "open": closes,
                    "high": [value + 1 for value in closes],
                    "low": [value - 1 for value in closes],
                    "close": closes,
                    "volume": [1000] * len(closes),
                })

            def latest_quote(self, symbol):
                return {
                    "price": 111.0,
                    "observed_at": "2026-01-09T20:00:00+00:00",
                    "source": "verification_quote",
                }

        original_provider = service._market_provider
        original_ticker = service._provider_ticker_for
        try:
            service._market_provider = lambda: Provider()
            service._provider_ticker_for = lambda _: "TEST.US"
            result = service.get_price_history("TEST", "1M")
        finally:
            service._market_provider = original_provider
            service._provider_ticker_for = original_ticker

        assert result["summary"]["current_price"] == 111.0
        assert result["series_type"] == "provider_adjusted_history"
        assert result["summary"]["current_price_source"] == "verification_quote"

        # L'universo storico EODHD deve essere neutrale e deve escludere
        # automaticamente ETF/OTC quando vengono richieste azioni ordinarie.
        universe_provider = object.__new__(EODHDProvider)
        universe_provider.cache_dir = Path(temp_dir)
        universe_provider._request = lambda path, params=None: [
            {
                "Code": "AAA", "Name": "AAA Inc", "Exchange": "NYSE",
                "Currency": "USD", "Country": "USA", "Type": "Common Stock",
            },
            {
                "Code": "FUND", "Name": "Fund", "Exchange": "NYSE ARCA",
                "Currency": "USD", "Country": "USA", "Type": "ETF",
            },
            {
                "Code": "OTC1", "Name": "OTC", "Exchange": "PINK",
                "Currency": "USD", "Country": "USA", "Type": "Common Stock",
            },
        ]
        neutral_symbols = universe_provider.exchange_symbols("US")
        assert [item["provider_ticker"] for item in neutral_symbols] == ["AAA.US"]

        # Backfill PIT: forti ribassi e rialzi rimangono coorti separate;
        # il finto split raw non diventa un evento e nessun peso viene promosso.
        periods = 500
        returns = np.full(periods, 0.03)
        returns[260] = -10.0
        returns[261] = -4.0
        returns[262] = 7.0
        returns[263] = 11.0
        adjusted = 100.0 * np.cumprod(1.0 + returns / 100.0)
        raw = adjusted.copy()
        raw[220:] = raw[220:] / 2.0
        dates = pd.date_range("2023-01-02", periods=periods, freq="B", tz="UTC")
        event_history = pd.DataFrame({
            "datetime": dates,
            "close": raw,
            "adjusted_close": adjusted,
            "volume": np.full(periods, 1_000_000.0),
        })
        benchmark_history = pd.DataFrame({
            "datetime": dates,
            "close": np.linspace(400.0, 450.0, periods),
            "adjusted_close": np.linspace(400.0, 450.0, periods),
            "volume": np.full(periods, 10_000_000.0),
        })
        learning_config = HistoricalLearningConfig(
            baseline_sessions=20,
            minimum_history_sessions=80,
            cooldown_sessions=3,
        )
        events = mine_point_in_time_events(
            "AAA",
            event_history,
            benchmark_history,
            provider_ticker="AAA.US",
            benchmark_ticker="SPY.US",
            as_of=dates[-1],
            model_version="verification-2.1",
            config=learning_config,
        )
        assert {item["event_side"] for item in events} == {"downside", "upside"}
        assert not any(
            pd.Timestamp(item["event_session"]) == dates[220]
            for item in events
        )
        assert all(len(item["outcomes"]) == 6 for item in events)

        class HistoricalProvider:
            def daily_history(self, symbol, outputsize=300, adjust="all"):
                frame = (
                    benchmark_history if symbol == "SPY.US" else event_history
                )
                return frame.tail(int(outputsize)).copy()

        historical_result = run_historical_backfill(
            [{
                "ticker": "AAA",
                "provider_ticker": "AAA.US",
                "benchmark_ticker": "SPY.US",
            }],
            HistoricalProvider(),
            years=2,
            as_of=dates[-1],
            model_version="verification-2.1",
            run_key="offline-verification-2.1",
            config=learning_config,
        )
        assert historical_result["status"] == "complete"
        assert historical_result["stats"]["events_by_side"]["downside"] > 0
        assert historical_result["stats"]["events_by_side"]["upside"] > 0
        assert historical_result["automatic_production_weight_changes"] is False

        print("RELEASE_VERIFICATION_OK")
        print(
            "checks=missing_data,narrative,adjusted_prices,no_lookahead,"
            "fx,snapshots,feedback,backup,filters,quote,missing_score,sec_cache"
            ",history_outcomes,watchlist_events,advanced_filters,"
            "neutral_universe,historical_pit,historical_directions,"
            "historical_immutability"
        )


if __name__ == "__main__":
    main()
