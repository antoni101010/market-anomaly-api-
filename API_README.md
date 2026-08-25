from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

import api.main as api_main
import service


class _ImmediateThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        self.target(*self.args)


def test_authenticated_background_backfill_and_directional_stats(monkeypatch):
    test_config = replace(
        service.CONFIG,
        api_key="integration-secret",
        historical_backfill_enabled=True,
        historical_backfill_max_years=12,
        historical_backfill_max_symbols=25,
    )
    monkeypatch.setattr(service, "CONFIG", test_config)
    monkeypatch.setattr(api_main, "CONFIG", test_config)
    monkeypatch.setattr(service.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(service, "_market_provider", lambda: object())
    monkeypatch.setattr(
        service,
        "_historical_backfill_universe",
        lambda tickers, limit: [{
            "ticker": "TEST",
            "provider_ticker": "TEST.US",
            "company": "Test Company",
            "benchmark_ticker": "SPY.US",
        }],
    )
    monkeypatch.setattr(
        service,
        "historical_learning_status",
        lambda run_id=None: {
            "run_id": run_id or "historical-run-1",
            "status": "done",
            "checkpoint": {"next_symbol_index": 1},
        },
    )

    captured = {}

    def fake_run(universe, provider, **kwargs):
        captured.update(kwargs)
        captured["universe"] = universe
        return {
            "run_id": "historical-run-1",
            "status": "done",
            "symbols_total": 1,
            "symbols_processed": 1,
            "symbols_failed": 0,
            "events_inserted": 7,
            "downside_events": 5,
            "upside_events": 2,
            "automatic_production_weight_changes": False,
        }

    monkeypatch.setattr(service, "run_historical_backfill", fake_run)
    monkeypatch.setattr(
        service,
        "historical_learning_stats",
        lambda: {
            "events_total": 7,
            "directions": {
                "downside": {"events": 5, "mean_relative_return_pct": 1.2},
                "upside": {"events": 2, "mean_relative_return_pct": -0.4},
            },
        },
    )
    with service._historical_backfill_lock:
        service._historical_backfill_state.clear()
        service._historical_backfill_state.update({"status": "idle"})

    client = TestClient(api_main.app)

    unauthorized = client.post(
        "/api/historical-backfill",
        json={"years": 5, "tickers": ["TEST.US"]},
    )
    assert unauthorized.status_code == 401

    started = client.post(
        "/api/historical-backfill",
        headers={"X-API-Key": "integration-secret"},
        json={
            "years": 5,
            "as_of": "2026-08-01",
            "tickers": ["TEST.US"],
            "limit": 1,
            "resume": True,
        },
    )
    assert started.status_code == 200
    assert started.json()["ok"] is True
    assert started.json()["automatic_production_weight_changes"] is False
    assert captured["years"] == 5
    assert captured["resume"] is True
    assert captured["config"].primary_event_side == "downside"
    assert captured["config"].downside_return_threshold_pct < 0
    assert captured["config"].upside_return_threshold_pct > 0

    status = client.get(
        "/api/historical-backfill/status",
        headers={"X-API-Key": "integration-secret"},
    )
    assert status.status_code == 200
    assert status.json()["run_id"] == "historical-run-1"
    assert status.json()["automatic_production_weight_changes"] is False

    stats = client.get(
        "/api/historical-learning/stats",
        headers={"X-API-Key": "integration-secret"},
    )
    assert stats.status_code == 200
    payload = stats.json()
    assert payload["directions"]["downside"]["events"] == 5
    assert payload["directions"]["upside"]["events"] == 2
    assert payload["directions"]["downside"] != payload["directions"]["upside"]
    assert payload["automatic_production_weight_changes"] is False


def test_backfill_is_disabled_by_default(monkeypatch):
    test_config = replace(
        service.CONFIG,
        api_key="integration-secret",
        historical_backfill_enabled=False,
    )
    monkeypatch.setattr(service, "CONFIG", test_config)
    monkeypatch.setattr(api_main, "CONFIG", test_config)

    response = TestClient(api_main.app).post(
        "/api/historical-backfill",
        headers={"X-API-Key": "integration-secret"},
        json={"years": 1, "tickers": ["TEST.US"]},
    )

    assert response.status_code == 503
    assert "disabilitato" in response.json()["detail"].lower()
