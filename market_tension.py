"""Global Market Tension Engine.

Il motore e' volutamente separato dalla shortlist di titoli in ribasso.
Costruisce un campione neutrale multi-mercato per le valutazioni e usa
benchmark regionali per misurare euforia dei prezzi e fragilita'.

Il risultato e' descrittivo/statistico: non e' una previsione di crollo,
una stima di rendimento o una raccomandazione di investimento.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


BENCHMARKS = {
    "Globale": "ACWI.US",
    "USA large cap": "SPY.US",
    "USA growth/tech": "QQQ.US",
    "USA small cap": "IWM.US",
    "Europa": "VGK.US",
    "Giappone": "EWJ.US",
    "Canada": "EWC.US",
    "Australia": "EWA.US",
    "Hong Kong": "EWH.US",
    "Emergenti": "EEM.US",
}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _piecewise(value: float | None, points: list[tuple[float, float]]) -> float | None:
    value = _finite(value)
    if value is None or not points:
        return None
    ordered = sorted((float(x), float(y)) for x, y in points)
    if value <= ordered[0][0]:
        return ordered[0][1]
    if value >= ordered[-1][0]:
        return ordered[-1][1]
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= value <= x1:
            span = x1 - x0
            if span == 0:
                return y1
            weight = (value - x0) / span
            return y0 + (y1 - y0) * weight
    return None


def _weighted_available(items: Iterable[tuple[float | None, float]]) -> float | None:
    pairs = [(float(v), float(w)) for v, w in items if _finite(v) is not None and w > 0]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight


def _region(exchange: str | None) -> str:
    code = str(exchange or "").upper()
    if code in {"US", "NYSE", "NASDAQ", "AMEX"}:
        return "USA"
    if code in {"LSE", "PA", "XETRA", "MI", "SW", "AS", "BR", "MC", "LS", "ST", "CO", "HE", "OL"}:
        return "Europa"
    if code in {"TO", "V"}:
        return "Canada"
    if code in {"TSE", "JP"}:
        return "Giappone"
    if code in {"HK"}:
        return "Hong Kong"
    if code in {"AU", "AX"}:
        return "Australia"
    if code in {"JSE"}:
        return "Sudafrica"
    return code or "Altro"


def company_valuation_pressure(row: dict) -> float | None:
    """Score 0-100: valori piu alti = multipli/tensione valutativa piu elevati."""
    pe = _finite(row.get("pe_ratio"))
    fpe = _finite(row.get("forward_pe"))
    ps = _finite(row.get("price_to_sales"))
    evs = _finite(row.get("ev_to_sales"))
    fcf = _finite(row.get("fcf_yield_pct"))

    # Multipli negativi/non economicamente interpretabili non vengono premiati:
    # sono esclusi dal singolo multiplo, mentre FCF negativo e' trattato come
    # pressione elevata invece di sembrare "economico".
    pe_score = None if pe is None or pe <= 0 else _piecewise(
        pe, [(8, 10), (15, 30), (22, 50), (30, 68), (45, 86), (70, 100)]
    )
    fpe_score = None if fpe is None or fpe <= 0 else _piecewise(
        fpe, [(8, 10), (15, 30), (22, 50), (30, 68), (45, 86), (70, 100)]
    )
    ps_score = None if ps is None or ps < 0 else _piecewise(
        ps, [(0.5, 10), (1.5, 28), (3, 48), (5, 66), (9, 86), (15, 100)]
    )
    evs_score = None if evs is None or evs < 0 else _piecewise(
        evs, [(0.5, 10), (1.5, 28), (3, 48), (5, 66), (9, 86), (15, 100)]
    )
    fcf_score = None if fcf is None else _piecewise(
        fcf, [(-8, 100), (0, 90), (2, 72), (4, 52), (7, 30), (12, 12)]
    )

    return _weighted_available([
        (pe_score, 0.24),
        (fpe_score, 0.16),
        (ps_score, 0.22),
        (evs_score, 0.18),
        (fcf_score, 0.20),
    ])


def _benchmark_metrics(history: pd.DataFrame) -> dict | None:
    if history is None or history.empty or "close" not in history.columns:
        return None
    close = pd.to_numeric(history["close"], errors="coerce").dropna()
    if len(close) < 60:
        return None

    last = float(close.iloc[-1])
    ma50 = float(close.tail(min(50, len(close))).mean())
    ma200 = float(close.tail(min(200, len(close))).mean())
    high252 = float(close.tail(min(252, len(close))).max())
    base126 = float(close.iloc[-min(126, len(close))])
    ret126 = ((last / base126) - 1.0) * 100.0 if base126 > 0 else None
    above200 = ((last / ma200) - 1.0) * 100.0 if ma200 > 0 else None
    above50 = ((last / ma50) - 1.0) * 100.0 if ma50 > 0 else None
    distance_high = ((last / high252) - 1.0) * 100.0 if high252 > 0 else None

    returns = close.pct_change().dropna().tail(20)
    vol20 = float(returns.std(ddof=0) * math.sqrt(252) * 100.0) if len(returns) >= 10 else None

    euphoria = _weighted_available([
        (_piecewise(above200, [(-20, 5), (-5, 25), (0, 40), (10, 65), (20, 85), (35, 100)]), 0.34),
        (_piecewise(above50, [(-12, 8), (-3, 30), (0, 42), (6, 68), (12, 88), (20, 100)]), 0.18),
        (_piecewise(distance_high, [(-35, 5), (-20, 22), (-10, 48), (-3, 75), (0, 90), (5, 100)]), 0.22),
        (_piecewise(ret126, [(-30, 5), (-10, 22), (0, 40), (15, 65), (30, 84), (55, 100)]), 0.26),
    ])

    return {
        "last": round(last, 6),
        "above_200d_pct": None if above200 is None else round(above200, 2),
        "above_50d_pct": None if above50 is None else round(above50, 2),
        "distance_52w_high_pct": None if distance_high is None else round(distance_high, 2),
        "return_6m_pct": None if ret126 is None else round(ret126, 2),
        "volatility_20d_ann_pct": None if vol20 is None else round(vol20, 2),
        "above_200d": bool(last >= ma200),
        "euphoria_score": None if euphoria is None else round(_clamp(euphoria), 1),
    }


def calculate_market_tension(
    valuation_rows: list[dict],
    benchmark_histories: dict[str, pd.DataFrame],
    *,
    expected_valuation_rows: int,
    expected_benchmarks: int | None = None,
    source: str = "provider",
    observed_at: str | None = None,
) -> dict:
    """Pure scoring function used by production and tests."""
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    expected_benchmarks = expected_benchmarks or len(BENCHMARKS)

    scored_companies: list[dict] = []
    for raw in valuation_rows:
        score = company_valuation_pressure(raw)
        if score is None:
            continue
        scored_companies.append({
            "ticker": raw.get("ticker") or raw.get("fundamentals_symbol"),
            "company": raw.get("company") or raw.get("company_name"),
            "exchange": raw.get("exchange"),
            "region": _region(raw.get("exchange")),
            "market_cap": _finite(raw.get("market_cap")),
            "score": round(_clamp(score), 1),
        })

    regional_scores: dict[str, float] = {}
    for region in sorted({item["region"] for item in scored_companies}):
        values = [item["score"] for item in scored_companies if item["region"] == region]
        if values:
            regional_scores[region] = round(float(statistics.median(values)), 1)

    # La componente valutativa globale combina due letture:
    # - 40% region-balanced, per non lasciare che un singolo mercato domini;
    # - 60% ponderata per capitalizzazione, per rispettare il peso economico
    #   effettivo dei mercati azionari mondiali.
    valuation_region_balanced = (
        float(statistics.mean(regional_scores.values()))
        if regional_scores else None
    )
    cap_pairs = [
        (item["score"], item["market_cap"])
        for item in scored_companies
        if _finite(item.get("market_cap")) is not None and float(item["market_cap"]) > 0
    ]
    valuation_cap_weighted = None
    if cap_pairs:
        cap_total = sum(cap for _, cap in cap_pairs)
        valuation_cap_weighted = (
            sum(score * cap for score, cap in cap_pairs) / cap_total
            if cap_total > 0 else None
        )
    valuation_pressure = _weighted_available([
        (valuation_region_balanced, 0.40),
        (valuation_cap_weighted, 0.60),
    ])
    if valuation_pressure is not None:
        valuation_pressure = round(_clamp(valuation_pressure), 1)

    benchmark_rows: list[dict] = []
    for name, history in benchmark_histories.items():
        metrics = _benchmark_metrics(history)
        if metrics is None:
            continue
        benchmark_rows.append({"name": name, **metrics})

    euphoria_values = [item["euphoria_score"] for item in benchmark_rows if item.get("euphoria_score") is not None]
    euphoria_score = round(float(statistics.mean(euphoria_values)), 1) if euphoria_values else None

    breadth_pct = None
    if benchmark_rows:
        breadth_pct = round(
            sum(1 for item in benchmark_rows if item.get("above_200d")) / len(benchmark_rows) * 100.0,
            1,
        )
    vol_values = [item["volatility_20d_ann_pct"] for item in benchmark_rows if item.get("volatility_20d_ann_pct") is not None]
    volatility = float(statistics.mean(vol_values)) if vol_values else None
    momentum_values = [item["return_6m_pct"] for item in benchmark_rows if item.get("return_6m_pct") is not None]
    dispersion = float(statistics.pstdev(momentum_values)) if len(momentum_values) >= 2 else None

    # Fragilita': volatilita elevata e forte dispersione aumentano lo score.
    # Una breadth debole mentre l'euforia media e' alta e' una divergenza fragile.
    volatility_score = _piecewise(volatility, [(8, 12), (15, 30), (25, 55), (40, 80), (60, 100)])
    dispersion_score = _piecewise(dispersion, [(3, 12), (8, 35), (15, 58), (25, 80), (40, 100)])
    breadth_divergence = None
    if breadth_pct is not None and euphoria_score is not None:
        narrowness = 100.0 - breadth_pct
        breadth_divergence = _clamp(narrowness * (0.45 + euphoria_score / 180.0))
    fragility_score = _weighted_available([
        (volatility_score, 0.40),
        (dispersion_score, 0.25),
        (breadth_divergence, 0.35),
    ])
    if fragility_score is not None:
        fragility_score = round(_clamp(fragility_score), 1)

    global_score = _weighted_available([
        (valuation_pressure, 0.45),
        (euphoria_score, 0.35),
        (fragility_score, 0.20),
    ])
    if global_score is not None:
        global_score = round(_clamp(global_score), 1)

    valuation_coverage = _clamp(
        len(scored_companies) / max(1, int(expected_valuation_rows)) * 100.0
    )
    benchmark_coverage = _clamp(
        len(benchmark_rows) / max(1, int(expected_benchmarks)) * 100.0
    )
    coverage = round(0.65 * valuation_coverage + 0.35 * benchmark_coverage, 1)

    if global_score is None:
        level = "Non disponibile"
    elif global_score < 25:
        level = "Contenuta"
    elif global_score < 50:
        level = "Moderata"
    elif global_score < 70:
        level = "Elevata"
    elif global_score < 85:
        level = "Molto elevata"
    else:
        level = "Estrema"

    status = "complete"
    if valuation_pressure is None or euphoria_score is None or coverage < 65:
        status = "partial"
    if global_score is None:
        status = "unavailable"

    available_components = [
        name for name, value in (
            ("valutazioni", valuation_pressure),
            ("euforia prezzi", euphoria_score),
            ("fragilita", fragility_score),
        ) if value is not None
    ]
    if status == "unavailable":
        explanation = "Dati insufficienti per calcolare la tensione globale in modo affidabile."
    else:
        explanation = (
            f"Tensione statistica {level.lower()} calcolata su "
            f"{', '.join(available_components)}. "
            "Il valore descrive condizioni di mercato osservate e non stima quando o se avverra una correzione."
        )
        if status == "partial":
            explanation += " La copertura e parziale: il punteggio va interpretato con maggiore cautela."

    return {
        "observed_at": observed_at,
        "status": status,
        "score": global_score,
        "level": level,
        "valuation_pressure": valuation_pressure,
        "valuation_region_balanced": None if valuation_region_balanced is None else round(_clamp(valuation_region_balanced), 1),
        "valuation_cap_weighted": None if valuation_cap_weighted is None else round(_clamp(valuation_cap_weighted), 1),
        "price_euphoria": euphoria_score,
        "fragility": fragility_score,
        "coverage_pct": coverage,
        "valuation_coverage_pct": round(valuation_coverage, 1),
        "benchmark_coverage_pct": round(benchmark_coverage, 1),
        "valuation_companies": len(scored_companies),
        "regional_valuation": regional_scores,
        "benchmark_breadth_above_200d_pct": breadth_pct,
        "benchmark_details": benchmark_rows,
        "source": source,
        "data_delay_note": (
            "Indicatore basato su dati EOD e fondamentali del provider: non e un dato real-time. "
            "Orari e ritardi possono variare per mercato e fonte."
        ),
        "methodology_version": "market-tension-1.0",
        "explanation": explanation,
        "historical_warning": (
            "Relazioni storiche, confronti e backtest non garantiscono risultati futuri."
        ),
        "not_investment_advice": True,
    }


def collect_market_tension(provider, *, exchanges: tuple[str, ...], sample_per_exchange: int = 5) -> dict:
    """Collect real provider data and calculate a neutral multi-market snapshot."""
    valuation_rows: list[dict] = []
    neutral_sample = pd.DataFrame()

    if hasattr(provider, "screen_market_sample"):
        neutral_sample = provider.screen_market_sample(
            exchanges=exchanges,
            limit_per_exchange=max(1, int(sample_per_exchange)),
        )

    if neutral_sample is not None and not neutral_sample.empty:
        for _, row in neutral_sample.iterrows():
            asset_type = str(row.get("type") or row.get("Type") or "").lower()
            if any(token in asset_type for token in ("fund", "etf", "warrant", "preferred", "bond")):
                continue
            code = str(row.get("code") or row.get("Code") or "").strip().upper()
            exchange = str(row.get("exchange") or row.get("Exchange") or "US").strip().upper()
            if not code:
                continue
            symbol = code if "." in code else f"{code}.{exchange}"
            try:
                fundamentals = provider.fundamentals(symbol, max_age_hours=24)
            except Exception:
                continue
            fundamentals = dict(fundamentals or {})
            fundamentals["ticker"] = symbol
            fundamentals["exchange"] = fundamentals.get("exchange") or exchange
            fundamentals["company"] = fundamentals.get("company_name") or row.get("name") or code
            valuation_rows.append(fundamentals)

    histories: dict[str, pd.DataFrame] = {}
    for name, symbol in BENCHMARKS.items():
        try:
            histories[name] = provider.daily_history(symbol, outputsize=270)
        except Exception:
            continue

    expected_valuation_rows = max(1, len(exchanges) * max(1, int(sample_per_exchange)))
    return calculate_market_tension(
        valuation_rows,
        histories,
        expected_valuation_rows=expected_valuation_rows,
        expected_benchmarks=len(BENCHMARKS),
        source="eodhd" if provider.__class__.__name__.lower().startswith("eodhd") else provider.__class__.__name__,
    )
