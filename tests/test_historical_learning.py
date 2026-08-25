from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import model
import storage
from historical_learning import (
    HistoricalLearningConfig,
    historical_learning_stats,
    mine_point_in_time_events,
    run_historical_backfill,
)


def _history(
    *,
    periods: int = 500,
    downside_index: int = 260,
    include_raw_split: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    returns = rng.normal(0.02, 0.22, periods)
    returns[downside_index] = -10.0
    returns[downside_index + 1] = -4.0
    returns[downside_index + 2] = 6.0
    returns[downside_index + 3] = 10.0
    adjusted = 100.0 * np.cumprod(1.0 + returns / 100.0)
    raw = adjusted.copy()
    if include_raw_split:
        raw[220:] = raw[220:] / 2.0
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-02", periods=periods, freq="B", tz="UTC"),
        "close": raw,
        "adjusted_close": adjusted,
        "volume": np.full(periods, 1_000_000.0),
    })


def _benchmark(periods: int = 500) -> pd.DataFrame:
    returns = np.full(periods, 0.04)
    adjusted = 400.0 * np.cumprod(1.0 + returns / 100.0)
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-02", periods=periods, freq="B", tz="UTC"),
        "close": adjusted,
        "adjusted_close": adjusted,
        "volume": np.full(periods, 10_000_000.0),
    })


@pytest.fixture
def learning_config():
    return HistoricalLearningConfig(
        baseline_sessions=20,
        minimum_history_sessions=80,
        cooldown_sessions=3,
    )


def test_point_in_time_mining_separates_directions_and_uses_adjusted_prices(
    learning_config,
):
    prices = _history(include_raw_split=True)
    benchmark = _benchmark()
    events = mine_point_in_time_events(
        "ACME",
        prices,
        benchmark,
        as_of=prices.iloc[-1]["datetime"],
        config=learning_config,
        model_version="test-v1",
    )

    sides = {item["event_side"] for item in events}
    assert sides == {"downside", "upside"}
    assert all(item["event_return_pct"] < 0 for item in events if item["event_side"] == "downside")
    assert all(item["event_return_pct"] > 0 for item in events if item["event_side"] == "upside")
    assert all(item["is_primary_downside"] for item in events if item["event_side"] == "downside")
    assert not any(item["is_primary_downside"] for item in events if item["event_side"] == "upside")
    assert not any(
        pd.Timestamp(item["event_session"]) == prices.iloc[220]["datetime"]
        for item in events
    ), "the raw 2-for-1 split must not become a detected event"
    assert all(len(item["outcomes"]) == 6 for item in events)
    assert {o["horizon_sessions"] for o in events[0]["outcomes"]} == {1, 3, 7, 30, 90, 180}


def test_event_features_are_point_in_time_and_outcomes_include_recovery(
    learning_config,
):
    prices = _history()
    benchmark = _benchmark()
    changed_future = prices.copy()
    changed_future.loc[270:, "adjusted_close"] *= 1.7
    changed_future.loc[270:, "close"] *= 1.7

    original = mine_point_in_time_events(
        "ACME", prices, benchmark,
        as_of=prices.iloc[-1]["datetime"], config=learning_config,
        model_version="test-v1",
    )
    modified = mine_point_in_time_events(
        "ACME", changed_future, benchmark,
        as_of=prices.iloc[-1]["datetime"], config=learning_config,
        model_version="test-v1",
    )
    event_session = prices.iloc[260]["datetime"].isoformat()
    left = next(item for item in original if item["event_session"] == event_session)
    right = next(item for item in modified if item["event_session"] == event_session)

    assert left["features"] == right["features"]
    assert left["snapshot_hash"] == right["snapshot_hash"]
    one_session = next(o for o in left["outcomes"] if o["horizon_sessions"] == 1)
    three_sessions = next(o for o in left["outcomes"] if o["horizon_sessions"] == 3)
    assert one_session["max_drawdown_pct"] < 0
    assert one_session["max_adverse_excursion_pct"] == one_session["max_drawdown_pct"]
    assert one_session["recovered"] is False
    assert three_sessions["recovered"] is True
    assert three_sessions["recovery_sessions"] == 3
    assert three_sessions["relative_return_pct"] == pytest.approx(
        three_sessions["absolute_return_pct"] - three_sessions["benchmark_return_pct"]
    )


class _Provider:
    def __init__(self, histories, fail_once=None):
        self.histories = histories
        self.fail_once = set(fail_once or [])
        self.calls = []

    def daily_history(self, symbol, outputsize=300, adjust="all"):
        self.calls.append((symbol, adjust))
        if symbol in self.fail_once:
            self.fail_once.remove(symbol)
            raise RuntimeError(f"temporary failure for {symbol}")
        return self.histories[symbol].tail(outputsize).copy()


def test_backfill_is_idempotent_resumable_audited_and_immutable(
    tmp_path,
    monkeypatch,
    learning_config,
):
    db_path = tmp_path / "historical.db"
    monkeypatch.setattr(
        storage,
        "CONFIG",
        SimpleNamespace(
            db_path=str(db_path),
            backup_dir=str(tmp_path / "backups"),
            backup_retention=2,
            model_version="test-v1",
        ),
    )
    prices = _history()
    provider = _Provider(
        {"SPY": _benchmark(), "AAA": prices, "BBB": prices},
        fail_once={"BBB"},
    )
    weights_before = model.BACKTEST_WEIGHTS.copy()
    as_of = prices.iloc[-1]["datetime"]
    universe = ["AAA", "BBB"]

    first = run_historical_backfill(
        universe,
        provider,
        years=2,
        as_of=as_of,
        model_version="test-v1",
        run_key="repeatable-run",
        config=learning_config,
    )
    assert first["status"] == "partial"
    assert first["symbols_processed"] == 1
    assert first["symbols_failed"] == 1
    calls_after_first = list(provider.calls)

    resumed = run_historical_backfill(
        universe,
        provider,
        years=2,
        as_of=as_of,
        model_version="test-v1",
        run_key="repeatable-run",
        config=learning_config,
    )
    assert resumed["status"] == "complete"
    assert resumed["symbols_processed"] == 2
    resumed_calls = provider.calls[len(calls_after_first):]
    assert resumed_calls == [("SPY", "all"), ("BBB", "all")]
    assert ("AAA", "all") not in resumed_calls
    assert resumed["automatic_production_weight_changes"] is False
    assert model.BACKTEST_WEIGHTS == weights_before

    with sqlite3.connect(db_path) as connection:
        before = {
            "events": connection.execute(
                "SELECT COUNT(*) FROM historical_event_snapshots"
            ).fetchone()[0],
            "outcomes": connection.execute(
                "SELECT COUNT(*) FROM historical_event_outcomes"
            ).fetchone()[0],
            "audit": connection.execute(
                "SELECT COUNT(*) FROM historical_learning_audit"
            ).fetchone()[0],
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM historical_learning_checkpoints WHERE status='complete'"
        ).fetchone()[0] == 2
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            connection.execute(
                "UPDATE historical_event_snapshots SET ticker='CHANGED'"
            )

    completed_replay = run_historical_backfill(
        universe,
        provider,
        years=2,
        as_of=as_of,
        model_version="test-v1",
        run_key="repeatable-run",
        config=learning_config,
    )
    assert completed_replay["status"] == "complete"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM historical_event_snapshots"
        ).fetchone()[0] == before["events"]
        assert connection.execute(
            "SELECT COUNT(*) FROM historical_event_outcomes"
        ).fetchone()[0] == before["outcomes"]
        assert connection.execute(
            "SELECT COUNT(*) FROM historical_learning_audit"
        ).fetchone()[0] == before["audit"]

    stats = historical_learning_stats()
    assert stats["events_by_side"]["downside"] > 0
    assert stats["events_by_side"]["upside"] > 0
    assert stats["primary_event_side"] == "downside"
    assert stats["automatic_production_weight_changes"] is False
