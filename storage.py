from __future__ import annotations
import json
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
    notes TEXT
);

CREATE TABLE IF NOT EXISTS latest_scan (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    scan_time TEXT NOT NULL,
    market_mode TEXT,
    payload_json TEXT NOT NULL
);
"""

def _connect():
    p = Path(CONFIG.db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    return con

def save_signals(df: pd.DataFrame) -> int:
    if df is None or df.empty: return 0
    now = datetime.now(timezone.utc).isoformat()
    n=0
    with _connect() as con:
        for _, r in df.iterrows():
            payload = {}
            for k,v in r.to_dict().items():
                if isinstance(v, (list,dict,str,int,float,bool)) or v is None:
                    try: json.dumps(v); payload[k]=v
                    except Exception: pass
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

def load_signals(limit=5000) -> pd.DataFrame:
    with _connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY signal_time DESC LIMIT ?",con,params=[int(limit)]
        )

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

# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def add_to_watchlist(ticker: str, company: str, price: float, anomaly_score: float, opportunity_score: float, notes: str = "") -> bool:
    with _connect() as con:
        try:
            con.execute("""
                INSERT INTO watchlist(ticker,company,added_at,price_at_add,anomaly_score_at_add,opportunity_score_at_add,notes)
                VALUES(?,?,?,?,?,?,?)
            """,(
                ticker.upper(), company, datetime.now(timezone.utc).isoformat(),
                _num(price), _num(anomaly_score), _num(opportunity_score), notes or ""
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

# ---------------------------------------------------------------------------
# Latest scan cache (replaces Streamlit's st.session_state for a stateless API)
# ---------------------------------------------------------------------------

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
    """Returns (scan_time:str|None, market_mode:str|None, records:list[dict])"""
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
