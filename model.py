import math

LIVE_WEIGHTS = {
    "drawdown": 0.22, "rsi": 0.09, "volume": 0.11, "momentum": 0.15,
    "shock": 0.08, "market_relative": 0.13, "sector_relative": 0.12, "quality": 0.10,
}

BACKTEST_WEIGHTS = {
    "drawdown": 0.22, "rsi": 0.09, "volume": 0.11, "momentum": 0.15,
    "shock": 0.08, "market_relative": 0.13, "sector_relative": 0.12, "quality_pit": 0.10,
}

WEIGHT_KEYS = [
    "drawdown","rsi","volume","momentum","shock","market_relative","sector_relative","quality_pit"
]

FEATURE_MAP = {
    "drawdown":"score_drawdown", "rsi":"score_rsi", "volume":"score_volume",
    "momentum":"score_momentum", "shock":"score_shock",
    "market_relative":"score_market_relative", "sector_relative":"score_sector_relative",
    "quality_pit":"score_quality_pit",
}

def clamp(x, lo=0.0, hi=100.0):
    try:
        x=float(x)
        if math.isnan(x): return 0.0
        return max(lo,min(hi,x))
    except Exception:
        return 0.0

def normalize_weights(weights):
    vals={k:max(0.0,float(weights.get(k,0.0))) for k in WEIGHT_KEYS}
    total=sum(vals.values())
    if total<=0: return BACKTEST_WEIGHTS.copy()
    return {k:v/total for k,v in vals.items()}

def technical_components(t, spy60, sector60):
    rel_market=float(t["return_60d_pct"])-float(spy60)
    rel_sector=float(t["return_60d_pct"])-float(sector60)
    return {
        "score_drawdown":clamp(abs(min(float(t["drawdown_52w_pct"]),0))*1.65),
        "score_rsi":clamp((45-float(t["rsi14"]))*2.4),
        "score_volume":clamp((float(t["volume_ratio_20d"])-1.0)*40),
        "score_momentum":clamp(abs(min(float(t["return_60d_pct"]),0))*1.25+abs(min(float(t["return_20d_pct"]),0))*0.8),
        "score_shock":clamp(abs(min(float(t["worst_day_20d_pct"]),0))*5.0),
        "score_market_relative":clamp(abs(min(rel_market,0))*2.0),
        "score_sector_relative":clamp(abs(min(rel_sector,0))*2.2),
        "relative_60d_vs_spy_pct":rel_market,
        "relative_60d_vs_sector_pct":rel_sector,
    }

def backtest_score(row, weights=None):
    w=normalize_weights(weights or BACKTEST_WEIGHTS)
    return clamp(sum(clamp(row.get(FEATURE_MAP[k],0))*w[k] for k in WEIGHT_KEYS))

def live_score(components, quality_score, weights=None):
    w=weights or LIVE_WEIGHTS
    technical = (
        ("score_drawdown", "drawdown"),
        ("score_rsi", "rsi"),
        ("score_volume", "volume"),
        ("score_momentum", "momentum"),
        ("score_shock", "shock"),
        ("score_market_relative", "market_relative"),
        ("score_sector_relative", "sector_relative"),
    )
    weighted = sum(clamp(components[value_key]) * w[weight_key]
                   for value_key, weight_key in technical)
    total_weight = sum(w[weight_key] for _, weight_key in technical)

    # La qualità mancante non diventa né 50 né 0: viene semplicemente esclusa.
    # La completezza limita separatamente l'Opportunity tramite Confidence.
    try:
        quality = float(quality_score)
        quality_available = math.isfinite(quality)
    except (TypeError, ValueError):
        quality_available = False
        quality = 0.0

    if quality_available and w.get("quality", 0) > 0:
        weighted += clamp(quality) * w["quality"]
        total_weight += w["quality"]

    return clamp(weighted / total_weight if total_weight else 0.0)
