"""Point-in-time historical event mining for Market Anomaly.

This subsystem is deliberately separated from live scoring.  It records
downside (the primary product semantics) and upside events in distinct cohorts,
uses adjusted prices throughout, and only creates candidate learning evidence.
It never imports or mutates production model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from config import CONFIG
from storage import (
    append_historical_learning_audit,
    begin_historical_learning_run,
    finish_historical_learning_run,
    historical_learning_stats as _persisted_historical_learning_stats,
    load_historical_learning_status,
    save_historical_event_snapshot,
    update_historical_learning_checkpoint,
)


FEATURE_SCHEMA_VERSION = "historical-event-pit-v1"
PRIMARY_EVENT_SIDE = "downside"
AUTOMATIC_PRODUCTION_WEIGHT_CHANGES = False


@dataclass(frozen=True)
class HistoricalLearningConfig:
    horizons: tuple[int, ...] = (1, 3, 7, 30, 90, 180)
    baseline_sessions: int = 60
    minimum_history_sessions: int = 252
    downside_return_threshold_pct: float = -5.0
    upside_return_threshold_pct: float = 5.0
    zscore_threshold: float = 2.0
    cooldown_sessions: int = 5
    recovery_tolerance_pct: float = 0.0
    require_all_horizons: bool = True
    primary_event_side: str = PRIMARY_EVENT_SIDE
    price_adjustment: str = "all"
    benchmark_ticker: str = "SPY"

    def __post_init__(self) -> None:
        horizons = tuple(int(item) for item in self.horizons)
        if horizons != (1, 3, 7, 30, 90, 180):
            raise ValueError("Historical horizons must be 1,3,7,30,90,180 sessions.")
        if int(self.baseline_sessions) < 10:
            raise ValueError("baseline_sessions must be at least 10")
        if int(self.minimum_history_sessions) < int(self.baseline_sessions) + 1:
            raise ValueError("minimum_history_sessions must exceed baseline_sessions")
        if float(self.downside_return_threshold_pct) >= 0:
            raise ValueError("downside_return_threshold_pct must be negative")
        if float(self.upside_return_threshold_pct) <= 0:
            raise ValueError("upside_return_threshold_pct must be positive")
        if float(self.zscore_threshold) <= 0:
            raise ValueError("zscore_threshold must be positive")
        if int(self.cooldown_sessions) < 0:
            raise ValueError("cooldown_sessions cannot be negative")
        if self.primary_event_side != PRIMARY_EVENT_SIDE:
            raise ValueError("The primary historical model must remain downside.")
        if str(self.price_adjustment).lower() != "all":
            raise ValueError("Historical learning requires fully adjusted prices.")

    def content_hash(self) -> str:
        return _hash_payload(asdict(self))


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_payload(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_timestamp(value=None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _prepare_adjusted_history(
    frame: pd.DataFrame,
    *,
    as_of=None,
) -> tuple[pd.DataFrame, str]:
    """Return an ordered adjusted-close series, bounded by the knowledge date."""
    if frame is None or frame.empty:
        raise ValueError("Adjusted price history is empty.")
    if "datetime" not in frame.columns:
        raise ValueError("Adjusted price history requires a datetime column.")

    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
    if "adjusted_close" in data.columns:
        adjusted = pd.to_numeric(data["adjusted_close"], errors="coerce")
        fallback = (
            pd.to_numeric(data["close"], errors="coerce")
            if "close" in data.columns else adjusted
        )
        data["adjusted_price"] = adjusted.fillna(fallback)
        source = "adjusted_close"
    elif "close" in data.columns:
        # Providers are requested with adjust='all'.  Their canonical close is
        # therefore the adjusted series even when no duplicate column is sent.
        data["adjusted_price"] = pd.to_numeric(data["close"], errors="coerce")
        source = "provider_close_adjust_all"
    else:
        raise ValueError("Adjusted price history requires close or adjusted_close.")

    if "volume" in data.columns:
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    else:
        data["volume"] = np.nan
    data = (
        data.dropna(subset=["datetime", "adjusted_price"])
        .loc[lambda item: item["adjusted_price"] > 0]
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
    )
    if as_of is not None:
        data = data[data["datetime"] <= _utc_timestamp(as_of)]
    data = data[["datetime", "adjusted_price", "volume"]].reset_index(drop=True)
    if data.empty:
        raise ValueError("No adjusted observations exist on or before as_of.")
    return data, source


def _price_as_of(data: pd.DataFrame, session) -> float | None:
    eligible = data[data["datetime"] <= _utc_timestamp(session)]
    if eligible.empty:
        return None
    value = float(eligible.iloc[-1]["adjusted_price"])
    return value if math.isfinite(value) and value > 0 else None


def _return_pct(end: float, start: float) -> float:
    return (float(end) / float(start) - 1.0) * 100.0


def _past_return(data: pd.DataFrame, event_index: int, sessions: int) -> float | None:
    previous_index = event_index - int(sessions)
    if previous_index < 0:
        return None
    return _return_pct(
        data.iloc[event_index]["adjusted_price"],
        data.iloc[previous_index]["adjusted_price"],
    )


def _point_in_time_features(
    data: pd.DataFrame,
    benchmark: pd.DataFrame,
    event_index: int,
    event_return_pct: float,
    event_zscore: float,
    event_side: str,
    price_source: str,
) -> dict:
    """Calculate features from rows at or before event_index only."""
    known = data.iloc[: event_index + 1]
    event_session = known.iloc[-1]["datetime"]
    closes = known["adjusted_price"].astype(float)
    daily_returns = closes.pct_change().dropna()
    trailing_peak = float(closes.tail(252).max())
    benchmark_signal = _price_as_of(benchmark, event_session)
    benchmark_previous = _price_as_of(
        benchmark,
        data.iloc[event_index - 1]["datetime"],
    )
    benchmark_event_return = (
        _return_pct(benchmark_signal, benchmark_previous)
        if benchmark_signal is not None and benchmark_previous is not None
        else None
    )
    volume = known["volume"]
    volume_baseline = volume.iloc[-21:-1].dropna().mean() if len(volume) >= 21 else np.nan
    volume_ratio = (
        float(volume.iloc[-1] / volume_baseline)
        if pd.notna(volume.iloc[-1]) and pd.notna(volume_baseline) and volume_baseline > 0
        else None
    )
    features = {
        "feature_cutoff_session": event_session.isoformat(),
        "event_side": event_side,
        "primary_event_side": PRIMARY_EVENT_SIDE,
        "event_return_pct": float(event_return_pct),
        "event_zscore": float(event_zscore),
        "benchmark_event_return_pct": benchmark_event_return,
        "event_relative_to_benchmark_pct": (
            float(event_return_pct - benchmark_event_return)
            if benchmark_event_return is not None else None
        ),
        "drawdown_from_252_session_peak_pct": _return_pct(closes.iloc[-1], trailing_peak),
        "volatility_20_sessions_pct": (
            float(daily_returns.tail(20).std(ddof=1) * math.sqrt(252) * 100.0)
            if len(daily_returns.tail(20)) >= 2 else None
        ),
        "volatility_60_sessions_pct": (
            float(daily_returns.tail(60).std(ddof=1) * math.sqrt(252) * 100.0)
            if len(daily_returns.tail(60)) >= 2 else None
        ),
        "volume_ratio_20_sessions": volume_ratio,
        "adjusted_price_source": price_source,
        "observations_known_at_event": int(len(known)),
    }
    for sessions in (1, 3, 7, 30, 90, 180):
        features[f"trailing_return_{sessions}_sessions_pct"] = _past_return(
            data, event_index, sessions
        )
    return features


def _outcomes_for_event(
    data: pd.DataFrame,
    benchmark: pd.DataFrame,
    event_index: int,
    event_side: str,
    as_of: pd.Timestamp,
    config: HistoricalLearningConfig,
) -> list[dict]:
    signal_price = float(data.iloc[event_index]["adjusted_price"])
    previous_price = float(data.iloc[event_index - 1]["adjusted_price"])
    signal_session = data.iloc[event_index]["datetime"]
    benchmark_signal = _price_as_of(benchmark, signal_session)
    output: list[dict] = []

    for horizon in config.horizons:
        outcome_index = event_index + int(horizon)
        if outcome_index >= len(data):
            if config.require_all_horizons:
                return []
            continue
        outcome_session = data.iloc[outcome_index]["datetime"]
        if outcome_session > as_of:
            if config.require_all_horizons:
                return []
            continue
        path = data.iloc[event_index : outcome_index + 1]["adjusted_price"].astype(float)
        path_returns = (path / signal_price - 1.0) * 100.0
        outcome_price = float(path.iloc[-1])
        absolute_return = float(path_returns.iloc[-1])
        benchmark_outcome = _price_as_of(benchmark, outcome_session)
        benchmark_return = (
            _return_pct(benchmark_outcome, benchmark_signal)
            if benchmark_signal is not None and benchmark_outcome is not None
            else None
        )

        # Recovery is side-aware.  For the primary downside population it means
        # regaining the pre-shock adjusted close; upside controls use the exact
        # inverse (reversion to the pre-spike close) and are never mixed with it.
        tolerance = float(config.recovery_tolerance_pct) / 100.0
        post_event = path.iloc[1:]
        if event_side == "downside":
            recovered_positions = np.flatnonzero(
                post_event.to_numpy() >= previous_price * (1.0 - tolerance)
            )
            adverse = float(min(0.0, path_returns.min()))
        else:
            recovered_positions = np.flatnonzero(
                post_event.to_numpy() <= previous_price * (1.0 + tolerance)
            )
            # For an upside mean-reversion control, further upside is adverse.
            adverse = float(-max(0.0, path_returns.max()))

        recovered = len(recovered_positions) > 0
        outcome = {
            "horizon_sessions": int(horizon),
            "outcome_session": outcome_session.isoformat(),
            # This is the earliest session at which the fixed-horizon outcome
            # was knowable.  Keeping it independent of the later backfill date
            # makes replayed immutable outcomes byte-for-byte idempotent.
            "evaluated_as_of": outcome_session.isoformat(),
            "adjusted_outcome_price": outcome_price,
            "benchmark_adjusted_outcome_price": benchmark_outcome,
            "absolute_return_pct": absolute_return,
            "benchmark_return_pct": benchmark_return,
            "relative_return_pct": (
                float(absolute_return - benchmark_return)
                if benchmark_return is not None else None
            ),
            "max_drawdown_pct": float(min(0.0, path_returns.min())),
            "max_adverse_excursion_pct": adverse,
            "max_favorable_excursion_pct": float(max(0.0, path_returns.max())),
            "recovered": bool(recovered),
            "recovery_sessions": (
                int(recovered_positions[0]) + 1 if recovered else None
            ),
        }
        output.append(outcome)
    return output


def mine_point_in_time_events(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    provider_ticker: str | None = None,
    benchmark_ticker: str = "SPY",
    as_of=None,
    model_version: str | None = None,
    config: HistoricalLearningConfig | None = None,
) -> list[dict]:
    """Mine mature events using only information available at each event date.

    Each returned item is an immutable event dictionary with an ``outcomes``
    list.  Downside and upside keys are direction-specific by construction.
    """
    settings = config or HistoricalLearningConfig()
    knowledge_time = (
        _utc_timestamp(as_of)
        if as_of is not None
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    data, price_source = _prepare_adjusted_history(prices, as_of=knowledge_time)
    benchmark, _ = _prepare_adjusted_history(benchmark_prices, as_of=knowledge_time)
    if len(data) < settings.minimum_history_sessions:
        return []

    daily_return = data["adjusted_price"].pct_change() * 100.0
    baseline = daily_return.shift(1).rolling(
        settings.baseline_sessions,
        min_periods=settings.baseline_sessions,
    )
    baseline_mean = baseline.mean()
    baseline_std = baseline.std(ddof=1)
    zscore = (daily_return - baseline_mean) / baseline_std.replace(0.0, np.nan)
    zero_variance_shock = baseline_std.eq(0.0) & daily_return.ne(baseline_mean)
    zscore = zscore.mask(
        zero_variance_shock,
        np.sign(daily_return - baseline_mean) * 999.0,
    )

    version = str(model_version or CONFIG.model_version)
    display_ticker = str(ticker).strip().upper()
    provider_symbol = str(provider_ticker or display_ticker).strip().upper()
    benchmark_symbol = str(benchmark_ticker or settings.benchmark_ticker).strip().upper()
    config_hash = settings.content_hash()
    next_allowed = {"downside": 0, "upside": 0}
    events: list[dict] = []
    first_index = max(
        int(settings.minimum_history_sessions) - 1,
        int(settings.baseline_sessions) + 1,
    )

    for event_index in range(first_index, len(data)):
        event_return = float(daily_return.iloc[event_index])
        event_zscore = float(zscore.iloc[event_index])
        if not math.isfinite(event_return) or not math.isfinite(event_zscore):
            continue
        event_side = None
        if (
            event_return <= float(settings.downside_return_threshold_pct)
            and event_zscore <= -float(settings.zscore_threshold)
        ):
            event_side = "downside"
        elif (
            event_return >= float(settings.upside_return_threshold_pct)
            and event_zscore >= float(settings.zscore_threshold)
        ):
            event_side = "upside"
        if event_side is None or event_index < next_allowed[event_side]:
            continue

        outcomes = _outcomes_for_event(
            data, benchmark, event_index, event_side, knowledge_time, settings
        )
        if settings.require_all_horizons and len(outcomes) != len(settings.horizons):
            continue
        if not outcomes:
            continue
        event_session = data.iloc[event_index]["datetime"]
        features = _point_in_time_features(
            data, benchmark, event_index, event_return, event_zscore,
            event_side, price_source,
        )
        event_key = "|".join((
            display_ticker,
            event_session.strftime("%Y-%m-%d"),
            event_side,
            version,
            config_hash,
        ))
        event_id = _hash_payload(event_key)
        benchmark_signal = _price_as_of(benchmark, event_session)
        event = {
            "event_id": event_id,
            "event_key": event_key,
            "ticker": display_ticker,
            "provider_ticker": provider_symbol,
            "benchmark_ticker": benchmark_symbol,
            "event_session": event_session.isoformat(),
            "event_side": event_side,
            "is_primary_downside": event_side == PRIMARY_EVENT_SIDE,
            "signal_adjusted_price": float(data.iloc[event_index]["adjusted_price"]),
            "previous_adjusted_price": float(data.iloc[event_index - 1]["adjusted_price"]),
            "benchmark_adjusted_price": benchmark_signal,
            "event_return_pct": event_return,
            "event_zscore": event_zscore,
            "baseline_sessions": int(settings.baseline_sessions),
            "price_adjustment": settings.price_adjustment,
            "model_version": version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "config_hash": config_hash,
            "features": features,
            "outcomes": outcomes,
        }
        event["snapshot_hash"] = _hash_payload({
            key: value for key, value in event.items() if key != "outcomes"
        })
        for outcome in event["outcomes"]:
            outcome["event_id"] = event_id
            outcome["outcome_hash"] = _hash_payload(outcome)
        events.append(event)
        next_allowed[event_side] = event_index + int(settings.cooldown_sessions) + 1
    return events


def _normalise_universe(
    universe: pd.DataFrame | Iterable[dict | str],
    default_benchmark: str,
) -> list[dict]:
    if isinstance(universe, pd.DataFrame):
        records = universe.to_dict(orient="records")
    else:
        records = [
            {"ticker": item} if isinstance(item, str) else dict(item)
            for item in universe
        ]
    output: dict[str, dict] = {}
    for row in records:
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        if not ticker:
            continue
        output[ticker] = {
            "ticker": ticker,
            "provider_ticker": str(
                row.get("provider_ticker") or row.get("api_ticker") or ticker
            ).strip().upper(),
            "benchmark_ticker": str(
                row.get("benchmark_ticker") or default_benchmark
            ).strip().upper(),
        }
    return [output[key] for key in sorted(output)]


def _fetch_adjusted_history(provider, symbol: str, outputsize: int) -> pd.DataFrame:
    try:
        return provider.daily_history(
            symbol, outputsize=int(outputsize), adjust="all"
        )
    except TypeError:
        # Compatibility for providers whose canonical history is adjusted but
        # whose older signature does not expose the adjustment keyword.
        return provider.daily_history(symbol, outputsize=int(outputsize))


def run_historical_backfill(
    universe: pd.DataFrame | Iterable[dict | str],
    market_provider,
    *,
    years: int = 10,
    as_of=None,
    model_version: str | None = None,
    run_key: str | None = None,
    config: HistoricalLearningConfig | None = None,
    resume: bool = True,
) -> dict:
    """Run an idempotent, checkpointed universe backfill.

    The result is an evidence ledger only.  No path in this runner can promote
    candidate weights or modify the production scoring model.
    """
    settings = config or HistoricalLearningConfig()
    knowledge_time = (
        _utc_timestamp(as_of)
        if as_of is not None
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    version = str(model_version or CONFIG.model_version)
    symbols = _normalise_universe(universe, settings.benchmark_ticker)
    if not symbols:
        raise ValueError("Historical-learning universe is empty.")
    input_hash = _hash_payload(symbols)
    deterministic_key = run_key or _hash_payload({
        "as_of": knowledge_time.isoformat(),
        "config_hash": settings.content_hash(),
        "input_hash": input_hash,
        "model_version": version,
        "years": int(years),
    })
    run_id = "hl-" + _hash_payload(deterministic_key)[:24]
    state = begin_historical_learning_run(
        run_id,
        deterministic_key,
        as_of_session=knowledge_time.isoformat(),
        config_hash=settings.content_hash(),
        input_hash=input_hash,
        model_version=version,
        symbols_total=len(symbols),
        resume=resume,
    )
    actual_run_id = state["run_id"]
    if state["status"] == "complete":
        return {
            **state,
            "run_key": deterministic_key,
            "primary_event_side": PRIMARY_EVENT_SIDE,
            "automatic_production_weight_changes": False,
            "stats": historical_learning_stats(),
        }

    append_historical_learning_audit(
        actual_run_id,
        "run_started",
        details={
            "as_of": knowledge_time.isoformat(), "years": int(years),
            "symbols_total": len(symbols), "config_hash": settings.content_hash(),
            "input_hash": input_hash,
            "automatic_production_weight_changes": False,
        },
        idempotency_key=f"{actual_run_id}:run_started",
    )
    completed_checkpoints = {
        item["ticker"] for item in state.get("checkpoints", [])
        if item["status"] == "complete"
    } if resume else set()
    totals = {
        "symbols_processed": len(completed_checkpoints),
        "symbols_failed": 0,
        "events_inserted": int(state.get("events_inserted") or 0),
        "events_existing": int(state.get("events_existing") or 0),
        "outcomes_inserted": int(state.get("outcomes_inserted") or 0),
        "downside_events": int(state.get("downside_events") or 0),
        "upside_events": int(state.get("upside_events") or 0),
        "checkpoint": state.get("last_checkpoint"),
    }
    outputsize = max(
        int(settings.minimum_history_sessions) + max(settings.horizons) + 5,
        int(years) * 252 + int(settings.minimum_history_sessions),
    )
    history_cache: dict[str, pd.DataFrame] = {}

    for item in symbols:
        ticker = item["ticker"]
        if ticker in completed_checkpoints:
            continue
        inserted_for_ticker = 0
        outcomes_for_ticker = 0
        events_for_ticker = 0
        try:
            update_historical_learning_checkpoint(
                actual_run_id, ticker, status="running", run_totals=totals
            )
            benchmark_symbol = item["benchmark_ticker"]
            if benchmark_symbol not in history_cache:
                history_cache[benchmark_symbol] = _fetch_adjusted_history(
                    market_provider, benchmark_symbol, outputsize
                )
            prices = _fetch_adjusted_history(
                market_provider, item["provider_ticker"], outputsize
            )
            events = mine_point_in_time_events(
                ticker,
                prices,
                history_cache[benchmark_symbol],
                provider_ticker=item["provider_ticker"],
                benchmark_ticker=benchmark_symbol,
                as_of=knowledge_time,
                model_version=version,
                config=settings,
            )
            events_for_ticker = len(events)
            last_session = None
            for event in events:
                last_session = event["event_session"]
                saved = save_historical_event_snapshot(
                    {key: value for key, value in event.items() if key != "outcomes"},
                    event["outcomes"],
                    run_id=actual_run_id,
                )
                if saved["event_inserted"]:
                    inserted_for_ticker += 1
                    totals["events_inserted"] += 1
                else:
                    totals["events_existing"] += 1
                outcomes_for_ticker += int(saved["outcomes_inserted"])
                totals["outcomes_inserted"] += int(saved["outcomes_inserted"])
                totals[f"{event['event_side']}_events"] += 1
                append_historical_learning_audit(
                    actual_run_id,
                    "event_observed",
                    ticker=ticker,
                    event_key=event["event_key"],
                    details={
                        "event_id": event["event_id"],
                        "event_side": event["event_side"],
                        "event_session": event["event_session"],
                        "snapshot_hash": event["snapshot_hash"],
                    },
                    idempotency_key=f"{actual_run_id}:event:{event['event_id']}",
                )
            totals["symbols_processed"] += 1
            totals["checkpoint"] = ticker
            update_historical_learning_checkpoint(
                actual_run_id,
                ticker,
                status="complete",
                last_event_session=last_session,
                events_seen=events_for_ticker,
                events_inserted=inserted_for_ticker,
                outcomes_inserted=outcomes_for_ticker,
                run_totals=totals,
            )
        except Exception as error:
            totals["symbols_failed"] += 1
            totals["checkpoint"] = ticker
            update_historical_learning_checkpoint(
                actual_run_id,
                ticker,
                status="failed",
                events_seen=events_for_ticker,
                events_inserted=inserted_for_ticker,
                outcomes_inserted=outcomes_for_ticker,
                error_message=str(error),
                run_totals=totals,
            )
            append_historical_learning_audit(
                actual_run_id,
                "symbol_failed",
                ticker=ticker,
                details={"error": str(error)},
                idempotency_key=f"{actual_run_id}:failed:{ticker}",
            )

    if totals["symbols_processed"] == 0 and totals["symbols_failed"]:
        final_status = "failed"
    elif totals["symbols_failed"]:
        final_status = "partial"
    else:
        final_status = "complete"
    final = finish_historical_learning_run(
        actual_run_id,
        status=final_status,
        totals=totals,
        error_message=(
            f"{totals['symbols_failed']} symbol(s) failed"
            if totals["symbols_failed"] else None
        ),
    )
    append_historical_learning_audit(
        actual_run_id,
        "run_finished",
        details={"status": final_status, **totals},
        idempotency_key=f"{actual_run_id}:run_finished:{final_status}",
    )
    return {
        **final,
        "run_key": deterministic_key,
        "primary_event_side": PRIMARY_EVENT_SIDE,
        "automatic_production_weight_changes": False,
        "stats": historical_learning_stats(),
    }


def historical_learning_status(run_id: str | None = None) -> dict:
    return load_historical_learning_status(run_id)


def historical_learning_stats() -> dict:
    return _persisted_historical_learning_stats()


# Explicit compatibility name for callers that describe the subsystem rather
# than the operation.  Both names execute the same immutable backfill.
run_historical_learning_backfill = run_historical_backfill
