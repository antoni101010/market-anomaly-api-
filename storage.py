from __future__ import annotations
import json
import hashlib
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from config import CONFIG

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_time TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company TEXT,
    price REAL,
    anomaly_score REAL,
    opportunity_score REAL,
    recovery_potential REAL,
    value_trap_risk REAL,
    catalyst_risk REAL,
    quality_score REAL,
    payload_json TEXT,
    UNIQUE(signal_time, ticker)
);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_time ON signals(ticker, signal_time);

CREATE TABLE IF NOT EXISTS alert_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    min_opportunity REAL NOT NULL DEFAULT 70,
    min_anomaly REAL NOT NULL DEFAULT 60,
    max_value_trap REAL NOT NULL DEFAULT 50,
    max_catalyst_risk REAL NOT NULL DEFAULT 60,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    company TEXT,
    added_at TEXT NOT NULL,
    price_at_add REAL,
    anomaly_score_at_add REAL,
    opportunity_score_at_add REAL,
    catalyst_label_at_add TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS latest_scan (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    scan_time TEXT NOT NULL,
    market_mode TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    status TEXT NOT NULL,
    message TEXT,
    scan_limit INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS signal_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_time TEXT NOT NULL,
    ticker TEXT NOT NULL,
    provider_ticker TEXT,
    company TEXT,
    exchange_code TEXT,
    currency TEXT,
    price REAL,
    price_observed_at TEXT,
    price_source TEXT,
    benchmark_ticker TEXT,
    model_version TEXT NOT NULL,
    data_completeness REAL,
    payload_json TEXT NOT NULL,
    UNIQUE(snapshot_time, ticker, model_version)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_time
ON signal_snapshots(ticker, snapshot_time);

CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    horizon_sessions INTEGER NOT NULL,
    due_at TEXT NOT NULL,
    evaluated_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    signal_price REAL,
    outcome_price REAL,
    benchmark_signal_price REAL,
    benchmark_outcome_price REAL,
    absolute_return_pct REAL,
    benchmark_return_pct REAL,
    relative_return_pct REAL,
    max_drawdown_pct REAL,
    recovered INTEGER,
    recovery_sessions INTEGER,
    error_message TEXT,
    UNIQUE(snapshot_id, horizon_sessions),
    FOREIGN KEY(snapshot_id) REFERENCES signal_snapshots(id)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_status_due
ON signal_outcomes(status, due_at);

CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    anomaly_score REAL,
    valuation_score REAL,
    confidence_score REAL,
    opportunity_score REAL,
    value_trap_risk REAL,
    model_version TEXT NOT NULL,
    UNIQUE(observed_at, ticker, model_version)
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    snapshot_id INTEGER,
    feedback_type TEXT NOT NULL,
    note TEXT,
    FOREIGN KEY(snapshot_id) REFERENCES signal_snapshots(id)
);

CREATE TABLE IF NOT EXISTS scan_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    provider TEXT,
    scanner_mode TEXT,
    requested_limit INTEGER,
    light_candidates INTEGER DEFAULT 0,
    analyzed INTEGER DEFAULT 0,
    valid INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS legal_acceptances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    installation_hash TEXT NOT NULL,
    terms_version TEXT NOT NULL,
    privacy_version TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    app_version TEXT,
    platform TEXT,
    terms_accepted INTEGER NOT NULL CHECK(terms_accepted IN (0,1)),
    privacy_notice_acknowledged INTEGER NOT NULL CHECK(privacy_notice_acknowledged IN (0,1)),
    UNIQUE(installation_hash, terms_version)
);
CREATE INDEX IF NOT EXISTS idx_legal_acceptance_installation
ON legal_acceptances(installation_hash, accepted_at DESC);

CREATE TABLE IF NOT EXISTS market_tension_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL UNIQUE,
    score REAL,
    level TEXT,
    status TEXT NOT NULL,
    coverage_pct REAL,
    methodology_version TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_tension_observed
ON market_tension_snapshots(observed_at DESC);

CREATE TABLE IF NOT EXISTS historical_learning_runs (
    run_id TEXT PRIMARY KEY,
    run_key TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','complete','partial','failed')),
    as_of_session TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    primary_event_side TEXT NOT NULL DEFAULT 'downside'
        CHECK(primary_event_side='downside'),
    automatic_production_weight_changes INTEGER NOT NULL DEFAULT 0
        CHECK(automatic_production_weight_changes=0),
    symbols_total INTEGER NOT NULL DEFAULT 0,
    symbols_processed INTEGER NOT NULL DEFAULT 0,
    symbols_failed INTEGER NOT NULL DEFAULT 0,
    events_inserted INTEGER NOT NULL DEFAULT 0,
    events_existing INTEGER NOT NULL DEFAULT 0,
    outcomes_inserted INTEGER NOT NULL DEFAULT 0,
    downside_events INTEGER NOT NULL DEFAULT 0,
    upside_events INTEGER NOT NULL DEFAULT 0,
    last_checkpoint TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_historical_runs_started
ON historical_learning_runs(started_at DESC);

CREATE TABLE IF NOT EXISTS historical_learning_checkpoints (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','complete','failed')),
    last_event_session TEXT,
    events_seen INTEGER NOT NULL DEFAULT 0,
    events_inserted INTEGER NOT NULL DEFAULT 0,
    outcomes_inserted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    error_message TEXT,
    PRIMARY KEY(run_id,ticker),
    FOREIGN KEY(run_id) REFERENCES historical_learning_runs(run_id)
);

CREATE TABLE IF NOT EXISTS historical_event_snapshots (
    event_id TEXT PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    first_run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    provider_ticker TEXT NOT NULL,
    benchmark_ticker TEXT NOT NULL,
    event_session TEXT NOT NULL,
    event_side TEXT NOT NULL CHECK(event_side IN ('downside','upside')),
    is_primary_downside INTEGER NOT NULL
        CHECK(is_primary_downside IN (0,1)),
    signal_adjusted_price REAL NOT NULL CHECK(signal_adjusted_price>0),
    previous_adjusted_price REAL NOT NULL CHECK(previous_adjusted_price>0),
    benchmark_adjusted_price REAL,
    event_return_pct REAL NOT NULL,
    event_zscore REAL NOT NULL,
    baseline_sessions INTEGER NOT NULL,
    price_adjustment TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    features_json TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(
        (event_side='downside' AND is_primary_downside=1 AND event_return_pct<0)
        OR
        (event_side='upside' AND is_primary_downside=0 AND event_return_pct>0)
    ),
    FOREIGN KEY(first_run_id) REFERENCES historical_learning_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_historical_events_ticker_session
ON historical_event_snapshots(ticker,event_session);
CREATE INDEX IF NOT EXISTS idx_historical_events_side_session
ON historical_event_snapshots(event_side,event_session);

CREATE TABLE IF NOT EXISTS historical_event_outcomes (
    event_id TEXT NOT NULL,
    horizon_sessions INTEGER NOT NULL
        CHECK(horizon_sessions IN (1,3,7,30,90,180)),
    outcome_session TEXT NOT NULL,
    evaluated_as_of TEXT NOT NULL,
    adjusted_outcome_price REAL NOT NULL CHECK(adjusted_outcome_price>0),
    benchmark_adjusted_outcome_price REAL,
    absolute_return_pct REAL NOT NULL,
    benchmark_return_pct REAL,
    relative_return_pct REAL,
    max_drawdown_pct REAL NOT NULL,
    max_adverse_excursion_pct REAL NOT NULL,
    max_favorable_excursion_pct REAL NOT NULL,
    recovered INTEGER NOT NULL CHECK(recovered IN (0,1)),
    recovery_sessions INTEGER,
    outcome_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(event_id,horizon_sessions),
    FOREIGN KEY(event_id) REFERENCES historical_event_snapshots(event_id)
);
CREATE INDEX IF NOT EXISTS idx_historical_outcomes_horizon
ON historical_event_outcomes(horizon_sessions);

CREATE TABLE IF NOT EXISTS historical_learning_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    action TEXT NOT NULL,
    ticker TEXT,
    event_key TEXT,
    details_json TEXT NOT NULL,
    audit_hash TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES historical_learning_runs(run_id)
);

CREATE TRIGGER IF NOT EXISTS immutable_historical_event_snapshots_update
BEFORE UPDATE ON historical_event_snapshots
BEGIN SELECT RAISE(ABORT, 'historical event snapshots are immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_historical_event_snapshots_delete
BEFORE DELETE ON historical_event_snapshots
BEGIN SELECT RAISE(ABORT, 'historical event snapshots are immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_historical_event_outcomes_update
BEFORE UPDATE ON historical_event_outcomes
BEGIN SELECT RAISE(ABORT, 'historical event outcomes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_historical_event_outcomes_delete
BEFORE DELETE ON historical_event_outcomes
BEGIN SELECT RAISE(ABORT, 'historical event outcomes are immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_historical_learning_audit_update
BEFORE UPDATE ON historical_learning_audit
BEGIN SELECT RAISE(ABORT, 'historical learning audit is immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_historical_learning_audit_delete
BEFORE DELETE ON historical_learning_audit
BEGIN SELECT RAISE(ABORT, 'historical learning audit is immutable'); END;
"""

def _connect():
    p = Path(CONFIG.db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=5000")
    con.executescript(SCHEMA)
    # Migrazione compatibile con il database già presente su Render.
    watchlist_columns = {
        row[1] for row in con.execute("PRAGMA table_info(watchlist)").fetchall()
    }
    if "catalyst_label_at_add" not in watchlist_columns:
        con.execute(
            "ALTER TABLE watchlist ADD COLUMN catalyst_label_at_add TEXT"
        )
    return con


def create_database_backup() -> str | None:
    """Crea una copia SQLite consistente e conserva solo le ultime copie."""
    source_path = Path(CONFIG.db_path)
    if not source_path.exists():
        return None
    backup_dir = Path(CONFIG.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"market_anomaly_{timestamp}.db"

    with sqlite3.connect(source_path) as source, sqlite3.connect(destination) as target:
        source.backup(target)

    backups = sorted(
        backup_dir.glob("market_anomaly_*.db"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in backups[max(1, int(CONFIG.backup_retention)):]:
        old.unlink(missing_ok=True)
    return str(destination)

def save_signals(df: pd.DataFrame) -> int:
    if df is None or df.empty: return 0
    now = datetime.now(timezone.utc).isoformat()
    n=0
    with _connect() as con:
        for _, r in df.iterrows():
            payload = {
                key: _serializable(value)
                for key, value in r.to_dict().items()
            }
            con.execute("""
                INSERT OR IGNORE INTO signals(
                    signal_time,ticker,company,price,anomaly_score,opportunity_score,
                    recovery_potential,value_trap_risk,catalyst_risk,quality_score,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,(
                now,str(r.get('ticker','')),str(r.get('company','')),
                _num(r.get('last_close')),_num(r.get('anomaly_score')),_num(r.get('opportunity_score')),
                _num(r.get('recovery_potential')),_num(r.get('value_trap_risk')),
                _num(r.get('catalyst_risk')),_num(r.get('quality_score')),json.dumps(payload,default=str)
            ))
            if con.total_changes>n: n+=1
    return n


def _json_payload(row: dict) -> str:
    clean = {key: _serializable(value) for key, value in row.items()}
    return json.dumps(clean, default=str, ensure_ascii=False)


def _serializable(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _business_due_at(snapshot_time: str, sessions: int) -> str:
    start = pd.Timestamp(snapshot_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    return (start + pd.offsets.BDay(int(sessions))).isoformat()


def save_signal_snapshots(
    df: pd.DataFrame,
    model_version: str | None = None,
    snapshot_time: str | None = None,
) -> int:
    """Salva casi immutabili e prepara gli outcome senza riscrivere il passato."""
    if df is None or df.empty:
        return 0

    observed_at = snapshot_time or datetime.now(timezone.utc).isoformat()
    version = str(model_version or CONFIG.model_version)
    saved = 0

    with _connect() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO model_versions(version,created_at,status,notes)
            VALUES(?,?,?,?)
            """,
            (version, observed_at, "production", "Snapshot automatico"),
        )

        for _, series in df.iterrows():
            row = series.to_dict()
            ticker = str(row.get("ticker") or "").upper()
            if not ticker or row.get("error"):
                continue

            cursor = con.execute(
                """
                INSERT OR IGNORE INTO signal_snapshots(
                    snapshot_time,ticker,provider_ticker,company,exchange_code,
                    currency,price,price_observed_at,price_source,benchmark_ticker,
                    model_version,data_completeness,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    observed_at,
                    ticker,
                    str(row.get("provider_ticker") or ticker),
                    str(row.get("company") or ticker),
                    str(row.get("light_exchange") or row.get("exchange") or ""),
                    str(row.get("currency") or ""),
                    _num(row.get("last_close")),
                    row.get("price_observed_at"),
                    str(row.get("price_source") or "historical_close"),
                    str(row.get("benchmark_ticker") or row.get("sector_etf") or "SPY"),
                    version,
                    _num(row.get("confidence_score")),
                    _json_payload(row),
                ),
            )

            if cursor.rowcount != 1:
                continue

            snapshot_id = int(cursor.lastrowid)
            saved += 1

            con.execute(
                """
                INSERT OR IGNORE INTO score_history(
                    observed_at,ticker,anomaly_score,valuation_score,
                    confidence_score,opportunity_score,value_trap_risk,model_version
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    observed_at,
                    ticker,
                    _num(row.get("anomaly_score")),
                    _num(row.get("valuation_score")),
                    _num(row.get("confidence_score")),
                    _num(row.get("opportunity_score")),
                    _num(row.get("value_trap_risk")),
                    version,
                ),
            )

            for horizon in (1, 3, 7, 30, 90, 180):
                con.execute(
                    """
                    INSERT OR IGNORE INTO signal_outcomes(
                        snapshot_id,horizon_sessions,due_at,status,signal_price
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        snapshot_id,
                        horizon,
                        _business_due_at(observed_at, horizon),
                        "pending",
                        _num(row.get("last_close")),
                    ),
                )

    return saved


def list_due_outcomes(limit: int = 250) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        rows = con.execute(
            """
            SELECT o.id,o.snapshot_id,o.horizon_sessions,o.due_at,o.signal_price,
                   s.ticker,s.provider_ticker,s.benchmark_ticker,s.snapshot_time,
                   s.payload_json
            FROM signal_outcomes o
            JOIN signal_snapshots s ON s.id=o.snapshot_id
            WHERE o.status='pending' AND o.due_at<=?
            ORDER BY o.due_at ASC
            LIMIT ?
            """,
            (now, max(1, int(limit))),
        ).fetchall()

    columns = [
        "outcome_id", "snapshot_id", "horizon_sessions", "due_at",
        "signal_price", "ticker", "provider_ticker", "benchmark_ticker",
        "snapshot_time", "payload_json",
    ]
    return [dict(zip(columns, row)) for row in rows]


def save_outcome(outcome_id: int, values: dict) -> None:
    with _connect() as con:
        con.execute(
            """
            UPDATE signal_outcomes SET
                evaluated_at=?,status=?,outcome_price=?,benchmark_signal_price=?,
                benchmark_outcome_price=?,absolute_return_pct=?,benchmark_return_pct=?,
                relative_return_pct=?,max_drawdown_pct=?,recovered=?,
                recovery_sessions=?,error_message=?
            WHERE id=?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                str(values.get("status") or "complete"),
                _num(values.get("outcome_price")),
                _num(values.get("benchmark_signal_price")),
                _num(values.get("benchmark_outcome_price")),
                _num(values.get("absolute_return_pct")),
                _num(values.get("benchmark_return_pct")),
                _num(values.get("relative_return_pct")),
                _num(values.get("max_drawdown_pct")),
                1 if values.get("recovered") else 0,
                values.get("recovery_sessions"),
                values.get("error_message"),
                int(outcome_id),
            ),
        )


def save_user_feedback(
    ticker: str,
    feedback_type: str,
    note: str = "",
    snapshot_id: int | None = None,
) -> int:
    allowed = {"useful", "possible_false_signal"}
    if feedback_type not in allowed:
        raise ValueError("Feedback non valido.")
    with _connect() as con:
        if snapshot_id is None:
            latest = con.execute(
                """
                SELECT id FROM signal_snapshots
                WHERE ticker=?
                ORDER BY snapshot_time DESC, id DESC
                LIMIT 1
                """,
                (str(ticker).upper(),),
            ).fetchone()
            snapshot_id = int(latest[0]) if latest else None

        cursor = con.execute(
            """
            INSERT INTO user_feedback(created_at,ticker,snapshot_id,feedback_type,note)
            VALUES(?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                str(ticker).upper(),
                snapshot_id,
                feedback_type,
                note or "",
            ),
        )
        return int(cursor.lastrowid)


def learning_summary() -> dict:
    with _connect() as con:
        snapshots = con.execute("SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
        pending = con.execute(
            "SELECT COUNT(*) FROM signal_outcomes WHERE status='pending'"
        ).fetchone()[0]
        completed = con.execute(
            "SELECT COUNT(*) FROM signal_outcomes WHERE status='complete'"
        ).fetchone()[0]
        row = con.execute(
            """
            SELECT AVG(relative_return_pct),AVG(max_drawdown_pct),
                   AVG(CASE WHEN recovered=1 THEN 1.0 ELSE 0.0 END)
            FROM signal_outcomes WHERE status='complete'
            """
        ).fetchone()
        horizon_rows = con.execute(
            """
            SELECT horizon_sessions,COUNT(*),AVG(absolute_return_pct),
                   AVG(relative_return_pct),
                   AVG(CASE WHEN relative_return_pct>0 THEN 1.0 ELSE 0.0 END),
                   AVG(max_drawdown_pct),
                   AVG(CASE WHEN recovered=1 THEN 1.0 ELSE 0.0 END)
            FROM signal_outcomes
            WHERE status='complete'
            GROUP BY horizon_sessions
            ORDER BY horizon_sessions
            """
        ).fetchall()
        feedback_rows = con.execute(
            "SELECT feedback_type,COUNT(*) FROM user_feedback GROUP BY feedback_type"
        ).fetchall()
    return {
        "snapshots": int(snapshots or 0),
        "outcomes_pending": int(pending or 0),
        "outcomes_completed": int(completed or 0),
        "mean_relative_return_pct": _num(row[0]) if row else None,
        "mean_max_drawdown_pct": _num(row[1]) if row else None,
        "recovery_rate": _num(row[2]) if row else None,
        "model_version": CONFIG.model_version,
        "automatic_production_retraining": False,
        "performance_by_horizon": [
            {
                "sessions": int(item[0]),
                "cases": int(item[1]),
                "mean_return_pct": _num(item[2]),
                "mean_relative_return_pct": _num(item[3]),
                "benchmark_win_rate": _num(item[4]),
                "mean_max_drawdown_pct": _num(item[5]),
                "recovery_rate": _num(item[6]),
            }
            for item in horizon_rows
        ],
        "feedback": {str(item[0]): int(item[1]) for item in feedback_rows},
        "promotion_rule": (
            "I pesi possono passare in produzione solo dopo backtest point-in-time, "
            "holdout, walk-forward e approvazione esplicita."
        ),
    }


def _canonical_json(value) -> str:
    return json.dumps(
        _serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_hash(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def begin_historical_learning_run(
    run_id: str,
    run_key: str,
    *,
    as_of_session: str,
    config_hash: str,
    input_hash: str,
    model_version: str,
    symbols_total: int,
    resume: bool = True,
) -> dict:
    """Create, or safely resume, one deterministic historical-learning run."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO historical_learning_runs(
                run_id,run_key,started_at,status,as_of_session,config_hash,
                input_hash,model_version,primary_event_side,
                automatic_production_weight_changes,symbols_total
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(run_id), str(run_key), now, "running", str(as_of_session),
                str(config_hash), str(input_hash), str(model_version),
                "downside", 0, max(0, int(symbols_total)),
            ),
        )
        row = con.execute(
            """
            SELECT run_id,status,as_of_session,config_hash,input_hash,model_version
            FROM historical_learning_runs WHERE run_key=?
            """,
            (str(run_key),),
        ).fetchone()
        if row is None:
            raise RuntimeError("Historical-learning run could not be created.")
        expected = (
            str(as_of_session), str(config_hash), str(input_hash),
            str(model_version),
        )
        if tuple(row[2:]) != expected:
            raise ValueError(
                "run_key already belongs to different immutable inputs/configuration"
            )
        actual_run_id = str(row[0])
        if bool(resume) and row[1] in {"partial", "failed"}:
            con.execute(
                """
                UPDATE historical_learning_runs
                SET status='running',finished_at=NULL,error_message=NULL
                WHERE run_id=?
                """,
                (actual_run_id,),
            )
    return load_historical_learning_status(actual_run_id)


def update_historical_learning_checkpoint(
    run_id: str,
    ticker: str,
    *,
    status: str,
    last_event_session: str | None = None,
    events_seen: int = 0,
    events_inserted: int = 0,
    outcomes_inserted: int = 0,
    error_message: str | None = None,
    run_totals: dict | None = None,
) -> None:
    allowed = {"pending", "running", "complete", "failed"}
    if status not in allowed:
        raise ValueError(f"Invalid historical checkpoint status: {status}")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute(
            """
            INSERT INTO historical_learning_checkpoints(
                run_id,ticker,status,last_event_session,events_seen,
                events_inserted,outcomes_inserted,updated_at,error_message
            ) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,ticker) DO UPDATE SET
                status=excluded.status,
                last_event_session=excluded.last_event_session,
                events_seen=excluded.events_seen,
                events_inserted=excluded.events_inserted,
                outcomes_inserted=excluded.outcomes_inserted,
                updated_at=excluded.updated_at,
                error_message=excluded.error_message
            """,
            (
                str(run_id), str(ticker).upper(), status, last_event_session,
                max(0, int(events_seen)), max(0, int(events_inserted)),
                max(0, int(outcomes_inserted)), now, error_message,
            ),
        )
        if run_totals is not None:
            con.execute(
                """
                UPDATE historical_learning_runs SET
                    symbols_processed=?,symbols_failed=?,events_inserted=?,
                    events_existing=?,outcomes_inserted=?,downside_events=?,
                    upside_events=?,last_checkpoint=?
                WHERE run_id=?
                """,
                (
                    max(0, int(run_totals.get("symbols_processed", 0))),
                    max(0, int(run_totals.get("symbols_failed", 0))),
                    max(0, int(run_totals.get("events_inserted", 0))),
                    max(0, int(run_totals.get("events_existing", 0))),
                    max(0, int(run_totals.get("outcomes_inserted", 0))),
                    max(0, int(run_totals.get("downside_events", 0))),
                    max(0, int(run_totals.get("upside_events", 0))),
                    str(ticker).upper(), str(run_id),
                ),
            )


def append_historical_learning_audit(
    run_id: str,
    action: str,
    *,
    ticker: str | None = None,
    event_key: str | None = None,
    details: dict | None = None,
    idempotency_key: str | None = None,
) -> bool:
    payload = details or {}
    stable_key = idempotency_key or _content_hash({
        "run_id": run_id,
        "action": action,
        "ticker": ticker,
        "event_key": event_key,
        "details": payload,
    })
    audit_hash = _content_hash({
        "idempotency_key": stable_key,
        "run_id": run_id,
        "action": action,
        "ticker": ticker,
        "event_key": event_key,
        "details": payload,
    })
    with _connect() as con:
        cursor = con.execute(
            """
            INSERT OR IGNORE INTO historical_learning_audit(
                idempotency_key,run_id,occurred_at,action,ticker,event_key,
                details_json,audit_hash
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                stable_key, str(run_id), datetime.now(timezone.utc).isoformat(),
                str(action), str(ticker).upper() if ticker else None,
                event_key, _canonical_json(payload), audit_hash,
            ),
        )
        return cursor.rowcount == 1


def save_historical_event_snapshot(
    event: dict,
    outcomes: list[dict],
    *,
    run_id: str,
) -> dict:
    """Atomically append one immutable PIT event and its mature outcomes.

    Replaying identical content is a no-op. A deterministic-key collision with
    different content is rejected rather than rewriting learned history.
    """
    event_id = str(event["event_id"])
    event_key = str(event.get("event_key") or event_id)
    features = event.get("features") or {}
    event_core = {
        key: event.get(key)
        for key in (
            "event_id", "event_key", "ticker", "provider_ticker",
            "benchmark_ticker", "event_session", "event_side",
            "is_primary_downside", "signal_adjusted_price",
            "previous_adjusted_price", "benchmark_adjusted_price",
            "event_return_pct", "event_zscore", "baseline_sessions",
            "price_adjustment", "model_version", "feature_schema_version",
            "config_hash",
        )
    }
    snapshot_hash = str(
        event.get("snapshot_hash") or _content_hash({**event_core, "features": features})
    )
    now = datetime.now(timezone.utc).isoformat()
    inserted_outcomes = 0
    with _connect() as con:
        existing = con.execute(
            "SELECT snapshot_hash FROM historical_event_snapshots WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if existing is not None and existing[0] != snapshot_hash:
            raise ValueError("immutable historical event collision")
        event_inserted = existing is None
        if event_inserted:
            con.execute(
                """
                INSERT INTO historical_event_snapshots(
                    event_id,event_key,first_run_id,ticker,provider_ticker,
                    benchmark_ticker,event_session,event_side,
                    is_primary_downside,signal_adjusted_price,
                    previous_adjusted_price,benchmark_adjusted_price,
                    event_return_pct,event_zscore,baseline_sessions,
                    price_adjustment,model_version,feature_schema_version,
                    config_hash,features_json,snapshot_hash,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, event_key, str(run_id),
                    str(event["ticker"]).upper(), str(event["provider_ticker"]),
                    str(event["benchmark_ticker"]), str(event["event_session"]),
                    str(event["event_side"]),
                    1 if event.get("is_primary_downside") else 0,
                    _num(event["signal_adjusted_price"]),
                    _num(event["previous_adjusted_price"]),
                    _num(event.get("benchmark_adjusted_price")),
                    _num(event["event_return_pct"]),
                    _num(event["event_zscore"]), int(event["baseline_sessions"]),
                    str(event["price_adjustment"]), str(event["model_version"]),
                    str(event["feature_schema_version"]),
                    str(event["config_hash"]), _canonical_json(features),
                    snapshot_hash, now,
                ),
            )

        for outcome in outcomes:
            horizon = int(outcome["horizon_sessions"])
            outcome_core = {
                key: outcome.get(key)
                for key in (
                    "event_id", "horizon_sessions", "outcome_session",
                    "evaluated_as_of", "adjusted_outcome_price",
                    "benchmark_adjusted_outcome_price", "absolute_return_pct",
                    "benchmark_return_pct", "relative_return_pct",
                    "max_drawdown_pct", "max_adverse_excursion_pct",
                    "max_favorable_excursion_pct", "recovered",
                    "recovery_sessions",
                )
            }
            outcome_hash = str(
                outcome.get("outcome_hash") or _content_hash(outcome_core)
            )
            existing_outcome = con.execute(
                """
                SELECT outcome_hash FROM historical_event_outcomes
                WHERE event_id=? AND horizon_sessions=?
                """,
                (event_id, horizon),
            ).fetchone()
            if existing_outcome is not None:
                if existing_outcome[0] != outcome_hash:
                    raise ValueError("immutable historical outcome collision")
                continue
            con.execute(
                """
                INSERT INTO historical_event_outcomes(
                    event_id,horizon_sessions,outcome_session,evaluated_as_of,
                    adjusted_outcome_price,benchmark_adjusted_outcome_price,
                    absolute_return_pct,benchmark_return_pct,relative_return_pct,
                    max_drawdown_pct,max_adverse_excursion_pct,
                    max_favorable_excursion_pct,recovered,recovery_sessions,
                    outcome_hash,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, horizon, str(outcome["outcome_session"]),
                    str(outcome["evaluated_as_of"]),
                    _num(outcome["adjusted_outcome_price"]),
                    _num(outcome.get("benchmark_adjusted_outcome_price")),
                    _num(outcome["absolute_return_pct"]),
                    _num(outcome.get("benchmark_return_pct")),
                    _num(outcome.get("relative_return_pct")),
                    _num(outcome["max_drawdown_pct"]),
                    _num(outcome["max_adverse_excursion_pct"]),
                    _num(outcome["max_favorable_excursion_pct"]),
                    1 if outcome.get("recovered") else 0,
                    outcome.get("recovery_sessions"), outcome_hash, now,
                ),
            )
            inserted_outcomes += 1
    return {
        "event_id": event_id,
        "event_inserted": event_inserted,
        "outcomes_inserted": inserted_outcomes,
    }


def finish_historical_learning_run(
    run_id: str,
    *,
    status: str,
    totals: dict,
    error_message: str | None = None,
) -> dict:
    if status not in {"complete", "partial", "failed"}:
        raise ValueError(f"Invalid historical run status: {status}")
    with _connect() as con:
        con.execute(
            """
            UPDATE historical_learning_runs SET
                finished_at=?,status=?,symbols_processed=?,symbols_failed=?,
                events_inserted=?,events_existing=?,outcomes_inserted=?,
                downside_events=?,upside_events=?,last_checkpoint=?,error_message=?
            WHERE run_id=?
            """,
            (
                datetime.now(timezone.utc).isoformat(), status,
                max(0, int(totals.get("symbols_processed", 0))),
                max(0, int(totals.get("symbols_failed", 0))),
                max(0, int(totals.get("events_inserted", 0))),
                max(0, int(totals.get("events_existing", 0))),
                max(0, int(totals.get("outcomes_inserted", 0))),
                max(0, int(totals.get("downside_events", 0))),
                max(0, int(totals.get("upside_events", 0))),
                totals.get("checkpoint"), error_message, str(run_id),
            ),
        )
    return load_historical_learning_status(run_id)


def load_historical_learning_status(run_id: str | None = None) -> dict:
    with _connect() as con:
        if run_id is None:
            row = con.execute(
                """
                SELECT * FROM historical_learning_runs
                ORDER BY started_at DESC,run_id DESC LIMIT 1
                """
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM historical_learning_runs WHERE run_id=?",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return {
                "status": "not_started",
                "primary_event_side": "downside",
                "automatic_production_weight_changes": False,
            }
        columns = [item[1] for item in con.execute(
            "PRAGMA table_info(historical_learning_runs)"
        ).fetchall()]
        output = dict(zip(columns, row))
        output["automatic_production_weight_changes"] = bool(
            output["automatic_production_weight_changes"]
        )
        checkpoint_rows = con.execute(
            """
            SELECT ticker,status,last_event_session,events_seen,events_inserted,
                   outcomes_inserted,updated_at,error_message
            FROM historical_learning_checkpoints WHERE run_id=?
            ORDER BY ticker
            """,
            (output["run_id"],),
        ).fetchall()
    output["checkpoints"] = [
        {
            "ticker": item[0], "status": item[1],
            "last_event_session": item[2], "events_seen": int(item[3]),
            "events_inserted": int(item[4]),
            "outcomes_inserted": int(item[5]), "updated_at": item[6],
            "error_message": item[7],
        }
        for item in checkpoint_rows
    ]
    return output


def historical_learning_stats() -> dict:
    with _connect() as con:
        side_rows = con.execute(
            """
            SELECT event_side,COUNT(*) FROM historical_event_snapshots
            GROUP BY event_side ORDER BY event_side
            """
        ).fetchall()
        horizon_rows = con.execute(
            """
            SELECT e.event_side,o.horizon_sessions,COUNT(*),
                   AVG(o.absolute_return_pct),AVG(o.relative_return_pct),
                   AVG(o.max_drawdown_pct),AVG(o.max_adverse_excursion_pct),
                   AVG(CASE WHEN o.recovered=1 THEN 1.0 ELSE 0.0 END)
            FROM historical_event_outcomes o
            JOIN historical_event_snapshots e ON e.event_id=o.event_id
            GROUP BY e.event_side,o.horizon_sessions
            ORDER BY e.event_side,o.horizon_sessions
            """
        ).fetchall()
        last_run = load_historical_learning_status()
    by_side = {"downside": 0, "upside": 0}
    by_side.update({str(item[0]): int(item[1]) for item in side_rows})
    return {
        "primary_event_side": "downside",
        "automatic_production_weight_changes": False,
        "events_total": sum(by_side.values()),
        "events_by_side": by_side,
        "performance_by_side_and_horizon": [
            {
                "event_side": item[0], "horizon_sessions": int(item[1]),
                "cases": int(item[2]), "mean_return_pct": _num(item[3]),
                "mean_relative_return_pct": _num(item[4]),
                "mean_max_drawdown_pct": _num(item[5]),
                "mean_adverse_excursion_pct": _num(item[6]),
                "recovery_rate": _num(item[7]),
            }
            for item in horizon_rows
        ],
        "last_run": last_run,
    }


def load_historical_events(
    *,
    event_side: str | None = None,
    ticker: str | None = None,
    limit: int = 500,
) -> list[dict]:
    if event_side is not None and event_side not in {"downside", "upside"}:
        raise ValueError("event_side must be 'downside' or 'upside'")
    where = []
    params: list = []
    if event_side is not None:
        where.append("event_side=?")
        params.append(event_side)
    if ticker is not None:
        where.append("ticker=?")
        params.append(str(ticker).upper())
    clause = " WHERE " + " AND ".join(where) if where else ""
    params.append(max(1, min(int(limit), 5000)))
    with _connect() as con:
        rows = con.execute(
            f"""
            SELECT event_id,event_key,ticker,provider_ticker,benchmark_ticker,
                   event_session,event_side,is_primary_downside,
                   signal_adjusted_price,previous_adjusted_price,
                   benchmark_adjusted_price,event_return_pct,event_zscore,
                   baseline_sessions,price_adjustment,model_version,
                   feature_schema_version,config_hash,features_json,
                   snapshot_hash,created_at
            FROM historical_event_snapshots{clause}
            ORDER BY event_session DESC,event_id DESC LIMIT ?
            """,
            params,
        ).fetchall()
        event_ids = [item[0] for item in rows]
        outcomes: dict[str, list[dict]] = {str(item): [] for item in event_ids}
        for start in range(0, len(event_ids), 800):
            chunk = event_ids[start:start + 800]
            placeholders = ",".join("?" for _ in chunk)
            outcome_rows = con.execute(
                f"""
                SELECT event_id,horizon_sessions,outcome_session,evaluated_as_of,
                       adjusted_outcome_price,benchmark_adjusted_outcome_price,
                       absolute_return_pct,benchmark_return_pct,
                       relative_return_pct,max_drawdown_pct,
                       max_adverse_excursion_pct,max_favorable_excursion_pct,
                       recovered,recovery_sessions
                FROM historical_event_outcomes
                WHERE event_id IN ({placeholders})
                ORDER BY event_id,horizon_sessions
                """,
                chunk,
            ).fetchall()
            for item in outcome_rows:
                outcomes[str(item[0])].append({
                    "horizon_sessions": int(item[1]),
                    "outcome_session": item[2], "evaluated_as_of": item[3],
                    "adjusted_outcome_price": _num(item[4]),
                    "benchmark_adjusted_outcome_price": _num(item[5]),
                    "absolute_return_pct": _num(item[6]),
                    "benchmark_return_pct": _num(item[7]),
                    "relative_return_pct": _num(item[8]),
                    "max_drawdown_pct": _num(item[9]),
                    "max_adverse_excursion_pct": _num(item[10]),
                    "max_favorable_excursion_pct": _num(item[11]),
                    "recovered": bool(item[12]),
                    "recovery_sessions": item[13],
                })
    output = []
    for item in rows:
        output.append({
            "event_id": item[0], "event_key": item[1], "ticker": item[2],
            "provider_ticker": item[3], "benchmark_ticker": item[4],
            "event_session": item[5], "event_side": item[6],
            "is_primary_downside": bool(item[7]),
            "signal_adjusted_price": _num(item[8]),
            "previous_adjusted_price": _num(item[9]),
            "benchmark_adjusted_price": _num(item[10]),
            "event_return_pct": _num(item[11]), "event_zscore": _num(item[12]),
            "baseline_sessions": int(item[13]), "price_adjustment": item[14],
            "model_version": item[15], "feature_schema_version": item[16],
            "config_hash": item[17], "features": json.loads(item[18]),
            "snapshot_hash": item[19], "created_at": item[20],
            "outcomes": outcomes.get(str(item[0]), []),
        })
    return output


def load_snapshot_history(limit: int = 500) -> list[dict]:
    """Restituisce gli snapshot immutabili insieme a tutti i loro esiti."""
    safe_limit = max(1, min(int(limit), 5000))
    with _connect() as con:
        snapshot_rows = con.execute(
            """
            SELECT id,snapshot_time,ticker,provider_ticker,company,
                   exchange_code,currency,price,price_observed_at,price_source,
                   benchmark_ticker,model_version,data_completeness,payload_json
            FROM signal_snapshots
            ORDER BY snapshot_time DESC,id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

        if not snapshot_rows:
            return []

        snapshot_ids = [int(row[0]) for row in snapshot_rows]
        outcome_rows = []
        # SQLite può avere un limite di 999 parametri: interroghiamo a blocchi.
        for start in range(0, len(snapshot_ids), 800):
            chunk = snapshot_ids[start:start + 800]
            placeholders = ",".join("?" for _ in chunk)
            outcome_rows.extend(con.execute(
                f"""
                SELECT snapshot_id,horizon_sessions,due_at,evaluated_at,status,
                       outcome_price,absolute_return_pct,benchmark_return_pct,
                       relative_return_pct,max_drawdown_pct,recovered,
                       recovery_sessions,error_message
                FROM signal_outcomes
                WHERE snapshot_id IN ({placeholders})
                ORDER BY snapshot_id,horizon_sessions
                """,
                chunk,
            ).fetchall())

    outcomes_by_snapshot: dict[int, list[dict]] = {
        snapshot_id: [] for snapshot_id in snapshot_ids
    }
    for row in outcome_rows:
        outcomes_by_snapshot[int(row[0])].append({
            "horizon_sessions": int(row[1]),
            "due_at": row[2],
            "evaluated_at": row[3],
            "status": row[4],
            "outcome_price": _num(row[5]),
            "absolute_return_pct": _num(row[6]),
            "benchmark_return_pct": _num(row[7]),
            "relative_return_pct": _num(row[8]),
            "max_drawdown_pct": _num(row[9]),
            "recovered": None if row[10] is None else bool(row[10]),
            "recovery_sessions": (
                int(row[11]) if row[11] is not None else None
            ),
            "error_message": row[12],
        })

    output = []
    for row in snapshot_rows:
        try:
            payload = json.loads(row[13] or "{}")
        except (TypeError, ValueError):
            payload = {}
        output.append({
            "snapshot_id": int(row[0]),
            "snapshot_time": row[1],
            "ticker": row[2],
            "provider_ticker": row[3],
            "company": row[4],
            "exchange": row[5],
            "currency": row[6],
            "price": _num(row[7]),
            "price_observed_at": row[8],
            "price_source": row[9],
            "benchmark_ticker": row[10],
            "model_version": row[11],
            "data_completeness": _num(row[12]),
            "payload": payload,
            "outcomes": outcomes_by_snapshot.get(int(row[0]), []),
        })
    return output


def begin_scan_run(run_id: str, provider: str, requested_limit: int) -> None:
    with _connect() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO scan_runs(
                run_id,started_at,provider,requested_limit,status
            ) VALUES(?,?,?,?,?)
            """,
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                provider,
                int(requested_limit),
                "running",
            ),
        )


def finish_scan_run(run_id: str, values: dict) -> None:
    with _connect() as con:
        con.execute(
            """
            UPDATE scan_runs SET
                finished_at=?,scanner_mode=?,light_candidates=?,analyzed=?,
                valid=?,failed=?,status=?,message=?
            WHERE run_id=?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                values.get("scanner_mode"),
                int(values.get("light_candidates") or 0),
                int(values.get("scanned") or 0),
                int(values.get("valid") or 0),
                int(values.get("failed") or 0),
                str(values.get("status") or "done"),
                str(values.get("message") or ""),
                run_id,
            ),
        )


def diagnostics() -> dict:
    path = Path(CONFIG.db_path)
    with _connect() as con:
        tables = {
            name: int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in (
                "signals", "signal_snapshots", "signal_outcomes",
                "watchlist", "scan_runs", "user_feedback",
                "legal_acceptances", "market_tension_snapshots",
            )
        }
    return {
        "database_path": str(path),
        "database_exists": path.exists(),
        "database_size_bytes": path.stat().st_size if path.exists() else 0,
        "tables": tables,
        "backup_count": len(list(Path(CONFIG.backup_dir).glob("market_anomaly_*.db")))
        if Path(CONFIG.backup_dir).exists()
        else 0,
    }

def load_signals(limit=5000) -> pd.DataFrame:
    with _connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY signal_time DESC LIMIT ?",con,params=[int(limit)]
        )

def save_legal_acceptance(
    installation_id: str,
    *,
    terms_version: str,
    privacy_version: str,
    app_version: str = "",
    platform: str = "",
    terms_accepted: bool = True,
    privacy_notice_acknowledged: bool = True,
) -> dict:
    """Store only a SHA-256 pseudonymous installation identifier."""
    installation_hash = hashlib.sha256(
        str(installation_id).strip().encode("utf-8")
    ).hexdigest()
    accepted_at = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute(
            """
            INSERT INTO legal_acceptances(
                installation_hash,terms_version,privacy_version,accepted_at,
                app_version,platform,terms_accepted,privacy_notice_acknowledged
            ) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(installation_hash,terms_version) DO UPDATE SET
              privacy_version=excluded.privacy_version,
              accepted_at=excluded.accepted_at,
              app_version=excluded.app_version,
              platform=excluded.platform,
              terms_accepted=excluded.terms_accepted,
              privacy_notice_acknowledged=excluded.privacy_notice_acknowledged
            """,
            (
                installation_hash, str(terms_version), str(privacy_version),
                accepted_at, str(app_version or ""), str(platform or ""),
                1 if terms_accepted else 0,
                1 if privacy_notice_acknowledged else 0,
            ),
        )
    return {
        "ok": True,
        "accepted_at": accepted_at,
        "terms_version": str(terms_version),
        "privacy_version": str(privacy_version),
    }


def delete_legal_installation_data(installation_id: str) -> int:
    installation_hash = hashlib.sha256(
        str(installation_id).strip().encode("utf-8")
    ).hexdigest()
    with _connect() as con:
        cursor = con.execute(
            "DELETE FROM legal_acceptances WHERE installation_hash=?",
            (installation_hash,),
        )
        return int(cursor.rowcount or 0)


def save_market_tension_snapshot(payload: dict) -> None:
    observed_at = str(payload.get("observed_at") or datetime.now(timezone.utc).isoformat())
    clean_payload = _json_payload(payload)
    with _connect() as con:
        con.execute(
            """
            INSERT INTO market_tension_snapshots(
                observed_at,score,level,status,coverage_pct,methodology_version,payload_json
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(observed_at) DO UPDATE SET
              score=excluded.score, level=excluded.level, status=excluded.status,
              coverage_pct=excluded.coverage_pct, methodology_version=excluded.methodology_version,
              payload_json=excluded.payload_json
            """,
            (
                observed_at, _num(payload.get("score")), str(payload.get("level") or ""),
                str(payload.get("status") or "unavailable"), _num(payload.get("coverage_pct")),
                str(payload.get("methodology_version") or "market-tension-1.0"), clean_payload,
            ),
        )


def load_market_tension_snapshot() -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT payload_json FROM market_tension_snapshots ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def market_tension_history(limit: int = 90) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT payload_json FROM market_tension_snapshots ORDER BY observed_at DESC LIMIT ?",
            (max(1, min(int(limit), 3650)),),
        ).fetchall()
    output = []
    for row in rows:
        try:
            item = json.loads(row[0])
            if isinstance(item, dict):
                output.append(item)
        except Exception:
            continue
    return output


def add_alert_rule(name,min_opportunity=70,min_anomaly=60,max_value_trap=50,max_catalyst_risk=60):
    with _connect() as con:
        con.execute("""
            INSERT INTO alert_rules(name,min_opportunity,min_anomaly,max_value_trap,max_catalyst_risk,enabled,created_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              min_opportunity=excluded.min_opportunity,
              min_anomaly=excluded.min_anomaly,
              max_value_trap=excluded.max_value_trap,
              max_catalyst_risk=excluded.max_catalyst_risk,
              enabled=1
        """,(name,float(min_opportunity),float(min_anomaly),float(max_value_trap),float(max_catalyst_risk),1,datetime.now(timezone.utc).isoformat()))

def list_alert_rules():
    with _connect() as con:
        return pd.read_sql_query("SELECT * FROM alert_rules ORDER BY id",con)

def set_alert_rule_enabled(rule_id, enabled):
    with _connect() as con:
        con.execute("UPDATE alert_rules SET enabled=? WHERE id=?",(1 if enabled else 0,int(rule_id)))

def delete_alert_rule(rule_id):
    with _connect() as con:
        con.execute("DELETE FROM alert_rules WHERE id=?",(int(rule_id),))

def _num(v):
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None

def add_to_watchlist(
    ticker: str,
    company: str,
    price: float,
    anomaly_score: float,
    opportunity_score: float,
    notes: str = "",
    catalyst_label: str = "",
) -> bool:
    with _connect() as con:
        try:
            con.execute("""
                INSERT INTO watchlist(
                    ticker,company,added_at,price_at_add,anomaly_score_at_add,
                    opportunity_score_at_add,catalyst_label_at_add,notes
                ) VALUES(?,?,?,?,?,?,?,?)
            """,(
                ticker.upper(), company, datetime.now(timezone.utc).isoformat(),
                _num(price), _num(anomaly_score), _num(opportunity_score),
                catalyst_label or "", notes or ""
            ))
            return True
        except sqlite3.IntegrityError:
            return False

def remove_from_watchlist(ticker: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.upper(),))

def list_watchlist() -> pd.DataFrame:
    with _connect() as con:
        return pd.read_sql_query("SELECT * FROM watchlist ORDER BY added_at DESC", con)

def is_in_watchlist(ticker: str) -> bool:
    with _connect() as con:
        row = con.execute("SELECT 1 FROM watchlist WHERE ticker=?", (ticker.upper(),)).fetchone()
    return row is not None

def save_latest_scan(df: pd.DataFrame, market_mode: str) -> None:
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    payload = json.dumps(records, default=str)
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("""
            INSERT INTO latest_scan(id,scan_time,market_mode,payload_json) VALUES(1,?,?,?)
            ON CONFLICT(id) DO UPDATE SET scan_time=excluded.scan_time, market_mode=excluded.market_mode, payload_json=excluded.payload_json
        """,(now, market_mode, payload))

def load_latest_scan():
    with _connect() as con:
        row = con.execute("SELECT scan_time, market_mode, payload_json FROM latest_scan WHERE id=1").fetchone()
    if not row:
        return None, None, []
    scan_time, market_mode, payload_json = row
    try:
        records = json.loads(payload_json)
    except Exception:
        records = []
    return scan_time, market_mode, records

def save_scan_state(state: dict) -> None:
    with _connect() as con:
        con.execute("""
            INSERT INTO scan_state(id,status,message,scan_limit,started_at,finished_at)
            VALUES(1,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              status=excluded.status,
              message=excluded.message,
              scan_limit=excluded.scan_limit,
              started_at=excluded.started_at,
              finished_at=excluded.finished_at
        """,(
            str(state.get("status", "idle")),
            str(state.get("message", "")),
            int(state.get("limit", 0) or 0),
            state.get("started_at"),
            state.get("finished_at"),
        ))

def load_scan_state() -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT status,message,scan_limit,started_at,finished_at FROM scan_state WHERE id=1"
        ).fetchone()
    if not row:
        return None
    return {
        "status": row[0],
        "message": row[1] or "",
        "limit": int(row[2] or 0),
        "started_at": row[3],
        "finished_at": row[4],
    }
