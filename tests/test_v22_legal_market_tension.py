from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
import service
import storage
from market_tension import BENCHMARKS, calculate_market_tension, company_valuation_pressure


def _history(*, start: float, end: float, periods: int = 270, wobble: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2025-08-01", periods=periods, freq="B", tz="UTC")
    base = np.linspace(start, end, periods)
    if wobble:
        base = base * (1.0 + np.sin(np.arange(periods) / 7.0) * wobble)
    return pd.DataFrame({"datetime": dates, "close": base})


def _valuation_rows(high: bool) -> list[dict]:
    rows = []
    exchanges = ["NASDAQ", "NYSE", "XETRA", "MI", "LSE", "TSE", "TO", "HK", "AU"]
    for idx, exchange in enumerate(exchanges):
        if high:
            values = {
                "pe_ratio": 48 + idx,
                "forward_pe": 40 + idx,
                "price_to_sales": 8.5 + idx * 0.2,
                "ev_to_sales": 8.0 + idx * 0.2,
                "fcf_yield_pct": 1.0,
            }
        else:
            values = {
                "pe_ratio": 13 + idx * 0.2,
                "forward_pe": 12 + idx * 0.2,
                "price_to_sales": 1.3 + idx * 0.03,
                "ev_to_sales": 1.4 + idx * 0.03,
                "fcf_yield_pct": 8.0,
            }
        rows.append({
            "ticker": f"T{idx}",
            "company": f"Company {idx}",
            "exchange": exchange,
            "market_cap": float(20_000_000_000 + idx * 5_000_000_000),
            **values,
        })
    return rows


def test_valuation_pressure_distinguishes_expensive_from_moderate_company():
    expensive = company_valuation_pressure({
        "pe_ratio": 60,
        "forward_pe": 50,
        "price_to_sales": 10,
        "ev_to_sales": 9,
        "fcf_yield_pct": 0.5,
    })
    moderate = company_valuation_pressure({
        "pe_ratio": 14,
        "forward_pe": 13,
        "price_to_sales": 1.5,
        "ev_to_sales": 1.6,
        "fcf_yield_pct": 8,
    })
    assert expensive is not None and moderate is not None
    assert expensive > moderate + 35


def test_global_market_tension_uses_neutral_valuation_and_benchmark_components():
    histories = {
        name: _history(start=100 + idx, end=150 + idx * 2, wobble=0.015)
        for idx, name in enumerate(BENCHMARKS)
    }
    high = calculate_market_tension(
        _valuation_rows(high=True),
        histories,
        expected_valuation_rows=9,
        expected_benchmarks=len(BENCHMARKS),
        source="synthetic-test",
        observed_at="2026-08-25T10:00:00+00:00",
    )
    low = calculate_market_tension(
        _valuation_rows(high=False),
        histories,
        expected_valuation_rows=9,
        expected_benchmarks=len(BENCHMARKS),
        source="synthetic-test",
        observed_at="2026-08-25T10:00:01+00:00",
    )

    assert high["status"] == "complete"
    assert high["coverage_pct"] == 100.0
    assert high["valuation_pressure"] > low["valuation_pressure"] + 35
    assert high["score"] > low["score"]
    assert high["valuation_region_balanced"] is not None
    assert high["valuation_cap_weighted"] is not None
    assert high["not_investment_advice"] is True
    assert "non stima" in high["explanation"].lower()


def test_legal_acceptance_is_versioned_hashed_and_deletable(tmp_path, monkeypatch):
    db_path = tmp_path / "v22.db"
    backend_config = replace(
        service.CONFIG,
        db_path=str(db_path),
        backup_dir=str(tmp_path / "backups"),
        api_key="v22-secret",
        legal_terms_version="2026-08-25-v1",
        legal_privacy_version="2026-08-25-v1",
    )
    monkeypatch.setattr(service, "CONFIG", backend_config)
    monkeypatch.setattr(api_main, "CONFIG", backend_config)
    monkeypatch.setattr(storage, "CONFIG", backend_config)

    client = TestClient(api_main.app)
    current = client.get("/api/legal/current")
    assert current.status_code == 200
    assert current.json()["product_positioning"] == "statistical_research_tool"
    assert current.json()["personalized_advice"] is False

    installation_id = "install-v22-random-example-123456"
    accepted = client.post(
        "/api/legal/acceptance",
        headers={"X-API-Key": "v22-secret"},
        json={
            "installation_id": installation_id,
            "terms_version": "2026-08-25-v1",
            "privacy_version": "2026-08-25-v1",
            "app_version": "2.2.0",
            "platform": "flutter-mobile",
            "terms_accepted": True,
            "privacy_notice_acknowledged": True,
        },
    )
    assert accepted.status_code == 200

    with storage._connect() as con:
        row = con.execute(
            "SELECT installation_hash,terms_version,privacy_version FROM legal_acceptances"
        ).fetchone()
    assert row is not None
    assert row[0] != installation_id
    assert len(row[0]) == 64
    assert row[1:] == ("2026-08-25-v1", "2026-08-25-v1")

    deleted = client.delete(
        f"/api/legal/installation/{installation_id}",
        headers={"X-API-Key": "v22-secret"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted_records"] == 1

    with storage._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM legal_acceptances").fetchone()[0] == 0
