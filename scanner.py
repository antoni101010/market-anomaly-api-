
import math
import pandas as pd

from indicators import (
    rsi14, drawdown_52w_pct, volume_ratio_20d,
    return_pct, volatility_20d_pct, worst_day_20d_pct
)
from fundamentals import quality_from_metrics, value_trap_risk
from providers.sec_edgar import SecEdgarProvider
from catalyst_engine import classify_catalysts, opportunity_score
from model import technical_components, live_score

def clamp(x, lo=0, hi=100):
    try:
        if math.isnan(float(x)):
            return 0.0
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

def base_technical(prices):
    return {
        "drawdown_52w_pct": drawdown_52w_pct(prices),
        "rsi14": rsi14(prices["close"]),
        "volume_ratio_20d": volume_ratio_20d(prices),
        "return_20d_pct": return_pct(prices["close"],20),
        "return_60d_pct": return_pct(prices["close"],60),
        "volatility_20d_pct": volatility_20d_pct(prices["close"]),
        "worst_day_20d_pct": worst_day_20d_pct(prices["close"]),
        "last_close": float(prices["close"].iloc[-1]),
    }

def score_one(t, quality, spy60, sector60):
    comps = technical_components(t, spy60, sector60)
    anomaly = live_score(comps, quality)
    return {
        "anomaly_score": round(anomaly, 1),
        "score_drawdown": round(comps["score_drawdown"], 1),
        "score_rsi": round(comps["score_rsi"], 1),
        "score_volume": round(comps["score_volume"], 1),
        "score_momentum": round(comps["score_momentum"], 1),
        "score_shock": round(comps["score_shock"], 1),
        "score_market_relative": round(comps["score_market_relative"], 1),
        "score_sector_relative": round(comps["score_sector_relative"], 1),
        "relative_60d_vs_spy_pct": round(comps["relative_60d_vs_spy_pct"], 2),
        "relative_60d_vs_sector_pct": round(comps["relative_60d_vs_sector_pct"], 2),
    }

def recovery_potential(anomaly, quality, trap, rel_sector):
    score = anomaly*0.48 + quality*0.27 + max(0,100-trap)*0.20 + clamp(abs(min(rel_sector,0))*1.2)*0.05
    return round(clamp(score),1)

def explanation(row):
    reasons = []
    if row["drawdown_52w_pct"] <= -30:
        reasons.append("forte drawdown dal massimo annuale")
    if row["relative_60d_vs_spy_pct"] <= -12:
        reasons.append("sottoperformance marcata rispetto a SPY")
    if row["relative_60d_vs_sector_pct"] <= -10:
        reasons.append("sottoperformance marcata rispetto al settore")
    if row["volume_ratio_20d"] >= 1.8:
        reasons.append("volumi anomali")
    if row["rsi14"] <= 32:
        reasons.append("RSI molto depresso")
    if row["quality_score"] >= 70:
        reasons.append("qualità fondamentale elevata")
    if row["value_trap_risk"] >= 70:
        reasons.append("rischio value trap elevato")
    elif row["value_trap_risk"] <= 40:
        reasons.append("rischio value trap relativamente contenuto")
    return "Il motore quantitativo segnala: " + (", ".join(reasons) if reasons else "anomalia moderata senza una causa quantitativa dominante") + "."

def scan_universe(universe, market_provider, include_sec=False, catalyst_top_n=5):
    sec = SecEdgarProvider() if include_sec else None

    symbols = list(universe["ticker"].astype(str).str.upper())
    sectors = list(universe["sector_etf"].astype(str).str.upper().unique())
    needed = sorted(set(symbols + sectors + ["SPY"]))

    histories = {}
    if hasattr(market_provider, "batch_daily_history"):
        try:
            histories = market_provider.batch_daily_history(needed, outputsize=300)
        except Exception:
            histories = {}

    def get_hist(sym):
        if sym in histories:
            return histories[sym]
        h = market_provider.daily_history(sym, outputsize=300)
        histories[sym] = h
        return h

    spy60 = base_technical(get_hist("SPY"))["return_60d_pct"]
    sector60 = {}
    for s in sectors:
        try:
            sector60[s] = base_technical(get_hist(s))["return_60d_pct"]
        except Exception:
            sector60[s] = spy60

    rows = []
    for _, u in universe.iterrows():
        ticker = str(u["ticker"]).upper()
        company = str(u["company"])
        sector = str(u["sector_etf"]).upper()
        try:
            prices = get_hist(ticker)
            if len(prices) < 65:
                raise ValueError("Storico insufficiente: servono almeno 65 sedute.")
            t = base_technical(prices)

            fm = {
                "revenue_growth_pct": None, "net_margin_pct": None,
                "liabilities_to_assets": None, "fcf_margin_pct": None,
                "approx_pe": None, "approx_ps": None,
            }

            if include_sec and sec is not None:
                try:
                    fm.update(sec.metrics(ticker, t["last_close"]))
                except Exception:
                    pass
            else:
                fm.update({
                    "revenue_growth_pct": float(u["demo_revenue_growth"]),
                    "net_margin_pct": float(u["demo_net_margin"]),
                    "liabilities_to_assets": float(u["demo_liab_assets"]),
                    "fcf_margin_pct": float(u["demo_fcf_margin"]),
                })

            quality = quality_from_metrics(fm)
            fm["quality_score"] = quality
            trap = value_trap_risk(fm, t)
            scores = score_one(t, quality, spy60, sector60.get(sector,spy60))
            recovery = recovery_potential(
                scores["anomaly_score"], quality, trap,
                scores["relative_60d_vs_sector_pct"]
            )

            row = {
                "ticker": ticker, "company": company, "sector_etf": sector,
                **t, **fm, **scores,
                "value_trap_risk": round(trap,1),
                "recovery_potential": recovery,
                "catalyst_label": "Non analizzato",
                "catalyst_risk": 50.0,
                "earnings_related": False,
                "catalyst_explanation": "Catalyst engine non ancora eseguito.",
                "catalyst_items": [],
                "recent_filings": [],
                "error": None,
            }
            row["explanation"] = explanation(row)
            rows.append(row)
        except Exception as e:
            rows.append({"ticker":ticker,"company":company,"sector_etf":sector,"error":str(e)})

    df = pd.DataFrame(rows)
    if "anomaly_score" not in df.columns:
        return df

    valid_idx = df[df["error"].isna()].sort_values("anomaly_score",ascending=False).head(int(catalyst_top_n)).index

    # Analizza i catalizzatori solo sui candidati migliori: risparmia crediti/API.
    for idx in valid_idx:
        ticker = df.at[idx,"ticker"]
        releases = []
        filings = []
        try:
            if hasattr(market_provider, "press_releases"):
                releases = market_provider.press_releases(ticker, limit=8)
        except Exception:
            releases = []
        if include_sec and sec is not None:
            try:
                filings = sec.recent_filings(ticker, limit=8)
            except Exception:
                filings = []

        cat = classify_catalysts(releases, filings)
        for k,v in cat.items():
            df.at[idx,k] = v

    # Opportunity Score: dopo aver analizzato i catalizzatori.
    opp = []
    for _, r in df.iterrows():
        if r.get("error") is not None and not pd.isna(r.get("error")):
            opp.append(None)
            continue
        opp.append(opportunity_score(
            r["anomaly_score"], r["quality_score"],
            r["value_trap_risk"], r["catalyst_risk"]
        ))
    df["opportunity_score"] = opp

    df = df.sort_values(["opportunity_score","anomaly_score"],ascending=False,na_position="last")
    return df.reset_index(drop=True)
