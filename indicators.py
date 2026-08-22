
import numpy as np
import pandas as pd

def rsi14(close: pd.Series) -> float:
    s = close.astype(float)
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val):
        return 100.0 if avg_loss.iloc[-1] == 0 else 50.0
    return float(val)

def drawdown_52w_pct(df):
    x = df.tail(252)
    high = float(x["high"].max())
    last = float(x["close"].iloc[-1])
    return (last / high - 1) * 100 if high else 0.0

def volume_ratio_20d(df):
    v = df["volume"].astype(float)
    if len(v) < 21:
        return 1.0
    baseline = v.iloc[-21:-1].mean()
    return float(v.iloc[-1] / baseline) if baseline else 1.0

def return_pct(close, periods):
    s = close.astype(float)
    if len(s) <= periods:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-1-periods] - 1) * 100)

def volatility_20d_pct(close):
    r = close.astype(float).pct_change().dropna().tail(20)
    return float(r.std(ddof=1) * np.sqrt(252) * 100) if len(r) >= 2 else 0.0

def worst_day_20d_pct(close):
    r = close.astype(float).pct_change().dropna().tail(20)
    return float(r.min() * 100) if len(r) else 0.0
