import math
import threading
from datetime import datetime, timezone

import pandas as pd

from indicators import (
    rsi14,
    drawdown_52w_pct,
    volume_ratio_20d,
    return_pct,
    volatility_20d_pct,
    worst_day_20d_pct,
)
from fundamentals import enrich_fundamental_scores, value_trap_risk
from providers.sec_edgar import SecEdgarProvider
from catalyst_engine import classify_catalysts, opportunity_score
from model import technical_components, live_score


_sec_provider_instance = None
_sec_provider_lock = threading.Lock()


def _sec_provider():
    global _sec_provider_instance
    with _sec_provider_lock:
        if _sec_provider_instance is None:
            _sec_provider_instance = SecEdgarProvider()
        return _sec_provider_instance


def clamp(x, lo=0, hi=100):
    try:
        if math.isnan(float(x)):
            return 0.0
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def overall_confidence(row: dict) -> float:
    """Coverage of fundamentals + price freshness + event context.

    This intentionally prevents a 100/100 label when the catalyst layer was
    not run or the quote/fundamentals are stale.
    """
    fundamental = row.get("fundamental_confidence_score", row.get("confidence_score"))
    try:
        fundamental = float(fundamental)
        if not math.isfinite(fundamental):
            fundamental = 0.0
    except (TypeError, ValueError):
        fundamental = 0.0

    validation = str(row.get("price_validation") or "").lower()
    observed = row.get("price_observed_at")
    age_hours = None
    try:
        ts = pd.Timestamp(observed)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_hours = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)
    except Exception:
        pass
    if validation == "provider_conflict":
        price_score = 15.0
    elif age_hours is None:
        price_score = 55.0
    elif age_hours <= 1.0:
        price_score = 92.0
    elif age_hours <= 24.0:
        price_score = 85.0
    elif age_hours <= 72.0:
        price_score = 72.0
    else:
        price_score = 40.0

    catalyst_label = str(row.get("catalyst_label") or "")
    if catalyst_label == "Non analizzato":
        catalyst_score = 25.0
    elif catalyst_label in {"Nessun catalizzatore disponibile", "Solo filing SEC rilevati"}:
        catalyst_score = 62.0
    elif catalyst_label:
        catalyst_score = 88.0
    else:
        catalyst_score = 35.0

    completeness = row.get("data_completeness") or {}
    groups = completeness.get("groups", {}) if isinstance(completeness, dict) else {}
    valuation_score = float(groups.get("valuation", 0.0) or 0.0)

    consistency = row.get("fundamental_consistency_score", 100.0)
    try:
        consistency = float(consistency)
        if not math.isfinite(consistency):
            consistency = 50.0
    except (TypeError, ValueError):
        consistency = 50.0

    score = (
        fundamental * 0.45
        + price_score * 0.15
        + catalyst_score * 0.20
        + valuation_score * 0.10
        + consistency * 0.10
    )
    validation_status = str(row.get("data_validation_status") or "").lower()
    if validation_status == "invalid":
        score = min(score, 45.0)
    elif validation_status == "secondary_listing":
        score = min(score, 55.0)
    return round(max(0.0, min(100.0, score)), 1)


def quote_matches_reference(quote: dict, reference) -> bool:
    """Rifiuta salti incompatibili con la chiusura EOD verificata.

    Quando il provider invia anche la chiusura precedente, la usiamo come
    ancora: così un vero forte movimento intraday non viene confuso con un
    errore di scala, valuta o corporate action.
    """
    try:
        price = float(quote.get("price"))
        reference_price = float(reference)
    except (AttributeError, TypeError, ValueError):
        return False

    if (
        not math.isfinite(price)
        or not math.isfinite(reference_price)
        or price <= 0
        or reference_price <= 0
    ):
        return False

    try:
        previous = float(quote.get("previous_close"))
    except (TypeError, ValueError):
        previous = None

    if previous is not None and math.isfinite(previous) and previous > 0:
        if abs(previous / reference_price - 1) > 0.35:
            return False
        return abs(price / previous - 1) <= 0.75

    return abs(price / reference_price - 1) <= 0.50


def _safe_number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _price_in_reporting_currency(price, listing_currency: str, reporting_currency: str):
    """Return a directly comparable price only when no FX guess is required."""
    price = _safe_number(price)
    if price is None:
        return None
    listing = str(listing_currency or "").upper()
    reporting = str(reporting_currency or "").upper()
    if not reporting or listing == reporting:
        return price
    if listing == "GBX" and reporting == "GBP":
        return price / 100.0
    if listing == "GBP" and reporting == "GBX":
        return price * 100.0
    return None


def reconcile_market_metrics(
    metrics: dict,
    technical: dict,
    *,
    provider_ticker: str,
    listing_currency: str,
    fx_rate: float | None = None,
) -> dict:
    """Cross-check price-dependent fundamentals before scoring/display.

    Provider fields are never trusted blindly.  Market cap is checked against
    shares × price on the primary listing, P/E against price/EPS and P/S
    against market-cap/revenue.  Large discrepancies are repaired when a
    deterministic same-currency calculation exists; otherwise the field is
    withheld and confidence is reduced.
    """
    out = dict(metrics)
    warnings: list[str] = []
    listing = str(listing_currency or "").upper()
    fundamentals_currency = str(
        out.get("fundamentals_currency")
        or out.get("reporting_currency")
        or listing
    ).upper()
    reporting = fundamentals_currency
    provider_symbol = str(provider_ticker or "").upper()
    primary = str(out.get("primary_ticker") or "").upper()
    is_primary = not primary or primary == provider_symbol

    price = _safe_number(
        technical.get("raw_eod_close")
        or technical.get("last_close")
        or technical.get("calculation_close")
    )
    comparable_price = _price_in_reporting_currency(price, listing, reporting)
    shares = _safe_number(out.get("shares_outstanding"))
    provider_cap = _safe_number(out.get("market_cap"))
    derived_cap = None
    if is_primary and comparable_price and shares and shares > 0:
        derived_cap = comparable_price * shares

    # Price-dependent fundamentals from a secondary listing can be expressed
    # in local depositary-receipt/quotation units that are not comparable with
    # the primary company's per-share fields.  Never present those as if they
    # were primary-listing valuation metrics.  Search/default analysis prefers
    # the primary listing; if a secondary listing is explicitly selected we
    # keep its technical price history but withhold ambiguous valuation fields.
    market_cap = provider_cap
    market_cap_source = "provider"
    if not is_primary:
        warnings.append(
            "Quotazione secondaria: multipli e capitalizzazione dipendenti dal prezzo sono esclusi finché non vengono riconciliati con la quotazione principale."
        )
        market_cap = None
        market_cap_source = "secondary_listing_withheld"
        for field in (
            "forward_pe", "ev_to_ebitda", "ev_to_sales",
            "price_to_book", "peg_ratio",
        ):
            out[field] = None
    if is_primary and derived_cap and derived_cap > 0:
        if provider_cap is None or provider_cap <= 0:
            market_cap = derived_cap
            market_cap_source = "price_x_shares"
        else:
            ratio = provider_cap / derived_cap
            if ratio < 0.50 or ratio > 2.0:
                market_cap = derived_cap
                market_cap_source = "price_x_shares_reconciled"
                warnings.append(
                    "Capitalizzazione provider incoerente con prezzo × azioni; usata la stima riconciliata."
                )
    if market_cap is not None and (market_cap <= 0 or market_cap > 50_000_000_000_000):
        warnings.append("Capitalizzazione fuori intervallo plausibile; dato escluso.")
        market_cap = None
        market_cap_source = "invalid"

    out["market_cap"] = market_cap
    out["market_cap_source"] = market_cap_source
    out["market_cap_derived"] = derived_cap
    out["market_cap_usd"] = None

    # FX is applied only after the listing/reporting unit has been normalised.
    try:
        fx = float(fx_rate) if fx_rate is not None else None
        if fx is not None and (not math.isfinite(fx) or fx <= 0):
            fx = None
    except (TypeError, ValueError):
        fx = None
    major_factor = 0.01 if reporting == "GBX" else 1.0
    if market_cap is not None:
        if reporting == "USD":
            out["market_cap_usd"] = market_cap * major_factor
        elif fx is not None and (reporting == listing or {reporting, listing} <= {"GBP", "GBX"}):
            out["market_cap_usd"] = market_cap * major_factor * fx

    eps = _safe_number(out.get("eps_ttm"))
    provider_pe = _safe_number(out.get("pe_ratio"))
    derived_pe = None
    if comparable_price and eps and eps > 0:
        derived_pe = comparable_price / eps
    pe = provider_pe if is_primary else None
    pe_source = "provider" if is_primary else "secondary_listing_withheld"
    if is_primary and derived_pe and derived_pe > 0:
        if provider_pe is None or provider_pe <= 0:
            pe, pe_source = derived_pe, "price_div_eps_ttm"
        else:
            ratio = provider_pe / derived_pe
            if ratio < 0.60 or ratio > 1.67:
                pe, pe_source = derived_pe, "price_div_eps_ttm_reconciled"
                warnings.append("P/E provider incoerente con prezzo/EPS TTM; usato il valore ricalcolato.")
    if pe is not None and (pe <= 0 or pe > 1000):
        warnings.append("P/E fuori intervallo plausibile; dato escluso.")
        pe, pe_source = None, "invalid"
    out["pe_ratio"] = pe
    out["pe_basis"] = "TTM ricalcolato" if "reconciled" in pe_source or pe_source == "price_div_eps_ttm" else out.get("pe_basis", "TTM/trailing")
    out["pe_source"] = pe_source
    out["pe_derived"] = derived_pe

    revenue = _safe_number(out.get("revenue_ttm"))
    provider_ps = _safe_number(out.get("price_to_sales"))
    derived_ps = market_cap / revenue if market_cap and revenue and revenue > 0 else None
    ps = provider_ps if is_primary else None
    ps_source = (
        str(out.get("price_to_sales_source") or "provider")
        if is_primary else "secondary_listing_withheld"
    )
    if is_primary and derived_ps and derived_ps > 0:
        if provider_ps is None or provider_ps <= 0:
            ps, ps_source = derived_ps, "market_cap_div_revenue_ttm"
        else:
            ratio = provider_ps / derived_ps
            if ratio < 0.60 or ratio > 1.67:
                ps, ps_source = derived_ps, "market_cap_div_revenue_ttm_reconciled"
                warnings.append("Prezzo/ricavi provider incoerente con capitalizzazione/ricavi; usato il valore ricalcolato.")
    if ps is not None and (ps <= 0 or ps > 500):
        warnings.append("Prezzo/ricavi fuori intervallo plausibile; dato escluso.")
        ps, ps_source = None, "invalid"
    out["price_to_sales"] = ps
    out["price_to_sales_source"] = ps_source

    fcf = _safe_number(out.get("free_cash_flow_ttm"))
    out["fcf_yield_pct"] = None
    if fcf is not None and market_cap and market_cap > 0:
        out["fcf_yield_pct"] = fcf / market_cap * 100.0

    out["listing_currency"] = listing
    out["currency"] = listing
    out["fundamentals_currency"] = fundamentals_currency
    # Backward-compatible alias used by the current mobile model.
    out["reporting_currency"] = fundamentals_currency
    out["is_primary_listing"] = bool(is_primary)
    out["data_validation_warnings"] = warnings
    if not is_primary:
        out["data_validation_status"] = "secondary_listing"
        out["fundamental_consistency_score"] = 35.0
    elif warnings:
        out["data_validation_status"] = "reconciled" if market_cap or pe or ps else "invalid"
        out["fundamental_consistency_score"] = max(35.0, 100.0 - len(warnings) * 18.0)
    else:
        out["data_validation_status"] = "ok"
        out["fundamental_consistency_score"] = 100.0
    return out


def base_technical(prices):
    last_row = prices.iloc[-1]
    observed_at = pd.Timestamp(last_row["datetime"])
    if observed_at.tzinfo is None:
        observed_at = observed_at.tz_localize("UTC")

    return {
        "drawdown_52w_pct": drawdown_52w_pct(prices),
        "rsi14": rsi14(prices["close"]),
        "volume_ratio_20d": volume_ratio_20d(prices),
        "return_20d_pct": return_pct(prices["close"], 20),
        "return_60d_pct": return_pct(prices["close"], 60),
        "volatility_20d_pct": volatility_20d_pct(prices["close"]),
        "worst_day_20d_pct": worst_day_20d_pct(prices["close"]),
        "last_close": float(prices["close"].iloc[-1]),
        "calculation_close": float(prices["close"].iloc[-1]),
        "raw_eod_close": (
            float(last_row["raw_close"])
            if "raw_close" in prices.columns and pd.notna(last_row.get("raw_close"))
            else float(prices["close"].iloc[-1])
        ),
        "price_observed_at": observed_at.isoformat(),
        "price_source": "historical_adjusted_close",
        "price_is_delayed": True,
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
        "score_market_relative": round(
            comps["score_market_relative"],
            1,
        ),
        "score_sector_relative": round(
            comps["score_sector_relative"],
            1,
        ),
        "relative_60d_vs_spy_pct": round(
            comps["relative_60d_vs_spy_pct"],
            2,
        ),
        "relative_60d_vs_sector_pct": round(
            comps["relative_60d_vs_sector_pct"],
            2,
        ),
    }


def recovery_potential(
    anomaly,
    quality,
    trap,
    rel_sector,
):
    components = [(clamp(anomaly), 0.48)]
    if quality is not None:
        components.append((clamp(quality), 0.27))
    if trap is not None:
        components.append((max(0, 100 - clamp(trap)), 0.20))
    if rel_sector is not None:
        components.append((clamp(abs(min(rel_sector, 0)) * 1.2), 0.05))

    total = sum(weight for _, weight in components)
    score = sum(value * weight for value, weight in components) / total

    return round(clamp(score), 1)


def explanation(row):
    reasons = []

    if row["drawdown_52w_pct"] <= -30:
        reasons.append(
            "forte distanza dal massimo annuale"
        )

    if row["relative_60d_vs_spy_pct"] <= -12:
        reasons.append(
            "forte sottoperformance rispetto al mercato"
        )

    if row["relative_60d_vs_sector_pct"] <= -10:
        reasons.append(
            "forte sottoperformance rispetto al settore"
        )

    if row["volume_ratio_20d"] >= 1.8:
        reasons.append(
            "volumi insolitamente elevati"
        )

    if row["rsi14"] <= 32:
        reasons.append(
            "pressione di vendita molto elevata"
        )

    quality = row.get("quality_score")
    trap = row.get("value_trap_risk")

    if quality is not None and quality >= 70:
        reasons.append(
            "qualità fondamentale elevata"
        )

    if trap is not None and trap >= 70:
        reasons.append(
            "rischio elevato che il calo sia giustificato"
        )

    elif trap is not None and trap <= 40:
        reasons.append(
            "rischio relativamente contenuto "
            "che il deterioramento sia strutturale"
        )

    detail = (
        ", ".join(reasons)
        if reasons
        else (
            "movimento insolito, ma senza "
            "un fattore quantitativo dominante"
        )
    )

    return (
        "Il motore ha rilevato: "
        + detail
        + "."
    )


def build_light_universe(
    market_provider,
    exchanges=("nasdaq", "nyse", "amex", "bats"),
    max_return_1d_pct=-8.0,
    min_avg_volume=50_000,
    min_price=1.0,
    min_market_cap=500_000_000,
    limit_per_exchange=100,
    max_rows=None,
):
    """LIGHT SCANNER globale.

    Preferisce il Bulk EOD extended del provider: in questo modo l'intero
    universo eleggibile viene controllato senza fare una richiesta per ogni
    singolo titolo. Il vecchio screener resta soltanto come fallback.
    """
    if hasattr(market_provider, "bulk_market_universe"):
        candidates = market_provider.bulk_market_universe(
            exchanges=exchanges,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            min_market_cap_usd=min_market_cap,
            max_rows=max_rows,
        )
        if candidates is None or candidates.empty:
            return pd.DataFrame(columns=["ticker", "company", "sector_etf"])

        rows = []
        for _, item in candidates.iterrows():
            ticker = str(item.get("ticker") or "").strip().upper()
            display_ticker = str(item.get("display_ticker") or ticker.split(".")[0]).strip().upper()
            if not ticker or not display_ticker:
                continue
            exchange = str(item.get("light_exchange") or ticker.rsplit(".", 1)[-1] or "US").strip().upper()
            sector_name = str(item.get("light_sector") or "").strip()
            row = item.to_dict()
            row.update({
                "ticker": ticker,
                "display_ticker": display_ticker,
                "company": str(item.get("company") or display_ticker),
                "sector_etf": _benchmark_for(exchange, sector_name),
                "benchmark_ticker": _market_benchmark(exchange),
                "light_exchange": exchange,
                "currency": _currency_for(exchange, item.get("currency")),
                "asset_type": str(item.get("asset_type") or "Common Stock"),
                "light_scanner_mode": "eodhd_bulk_global",
            })
            rows.append(row)
        return pd.DataFrame(rows)

    if not hasattr(market_provider, "screen_candidates"):
        return None

    candidates = market_provider.screen_candidates(
        exchanges=exchanges,
        max_return_1d_pct=max_return_1d_pct,
        min_avg_volume=max(200_000, int(min_avg_volume)),
        min_price=max(2.0, float(min_price)),
        min_market_cap=min_market_cap,
        limit_per_exchange=limit_per_exchange,
    )

    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["ticker", "company", "sector_etf"])

    rows = []
    for _, item in candidates.iterrows():
        asset_type = str(item.get("type") or item.get("Type") or item.get("asset_type") or "").strip()
        if any(excluded in asset_type.lower() for excluded in ("fund", "etf", "warrant", "preferred", "bond")):
            continue
        ticker = str(item.get("code", "")).strip().upper()
        if not ticker:
            continue
        exchange = str(item.get("exchange", "US")).strip().upper()
        api_ticker = ticker if "." in ticker else f"{ticker}.{exchange}"
        sector_name = str(item.get("sector") or "").strip()
        rows.append({
            "ticker": api_ticker,
            "display_ticker": ticker,
            "company": str(item.get("name", ticker)).strip(),
            "sector_etf": _benchmark_for(exchange, sector_name),
            "benchmark_ticker": _market_benchmark(exchange),
            "light_return_1d_pct": item.get("refund_1d_p"),
            "light_last_price": item.get("adjusted_close"),
            "light_market_cap": item.get("market_capitalization"),
            "light_volume": item.get("avgvol_200d"),
            "light_data_date": item.get("last_day_data_date"),
            "light_sector": item.get("sector"),
            "light_industry": item.get("industry"),
            "light_exchange": exchange,
            "currency": _currency_for(exchange, item.get("currency") or item.get("currency_symbol")),
            "country": item.get("country") or item.get("country_name"),
            "asset_type": asset_type or "Common Stock",
            "light_anomaly_score": max(0.0, min(100.0, abs(float(item.get("refund_1d_p") or 0.0)) * 4.0)),
            "light_scanner_mode": "eodhd_screener_fallback",
        })
    return pd.DataFrame(rows)


US_SECTOR_ETFS = {
    "technology": "XLK.US",
    "healthcare": "XLV.US",
    "financial": "XLF.US",
    "financial services": "XLF.US",
    "consumer cyclical": "XLY.US",
    "consumer defensive": "XLP.US",
    "communication services": "XLC.US",
    "industrials": "XLI.US",
    "energy": "XLE.US",
    "basic materials": "XLB.US",
    "real estate": "XLRE.US",
    "utilities": "XLU.US",
}


def _market_benchmark(exchange: str) -> str:
    code = str(exchange or "US").upper()
    if code in {"US", "NYSE", "NASDAQ", "AMEX", "BATS"}:
        return "SPY.US"
    if code in {
        "LSE", "L", "F", "XETRA", "PA", "MI", "SW", "AS",
        "BR", "MC", "LS", "ST", "CO", "HE", "OL",
    }:
        return "VGK.US"
    if code in {"TO", "V"}:
        return "EWC.US"
    if code in {"TSE", "JP"}:
        return "EWJ.US"
    if code in {"AU", "AX"}:
        return "EWA.US"
    if code in {"HK"}:
        return "EWH.US"
    if code in {"SHE", "SZSE", "SHG"}:
        return "MCHI.US"
    if code in {"JSE"}:
        return "EZA.US"
    return "ACWI.US"


def _currency_for(exchange: str, value) -> str:
    raw = str(value or "").strip().upper()
    symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
    if raw in symbols:
        return symbols[raw]
    if len(raw) == 3 and raw.isalpha():
        return raw
    code = str(exchange or "US").upper()
    mapping = {
        "US": "USD", "NYSE": "USD", "NASDAQ": "USD", "AMEX": "USD", "BATS": "USD",
        "LSE": "GBP", "TO": "CAD", "V": "CAD", "PA": "EUR",
        "XETRA": "EUR", "F": "EUR", "MI": "EUR", "AS": "EUR", "BR": "EUR",
        "MC": "EUR", "LS": "EUR", "SW": "CHF", "ST": "SEK",
        "CO": "DKK", "HE": "EUR", "OL": "NOK", "TSE": "JPY",
        "HK": "HKD", "AU": "AUD", "AX": "AUD", "JSE": "ZAR", "WAR": "PLN",
    }
    return mapping.get(code, "USD")


def _benchmark_for(exchange: str, sector: str) -> str:
    code = str(exchange or "US").upper()
    if code in {"US", "NYSE", "NASDAQ", "AMEX", "BATS"}:
        return US_SECTOR_ETFS.get(str(sector).lower(), "SPY.US")
    return _market_benchmark(code)


def scan_universe(
    universe,
    market_provider,
    include_sec=False,
    catalyst_top_n=5,
):
    sec = _sec_provider() if include_sec else None

    symbols = list(
        universe["ticker"]
        .astype(str)
        .str.upper()
    )

    sectors = list(
        universe["sector_etf"]
        .astype(str)
        .str.upper()
        .unique()
    )

    market_benchmarks = list(
        universe.get("benchmark_ticker", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    needed = sorted(
        set(
            symbols
            + sectors
            + market_benchmarks
            + ["SPY.US"]
        )
    )

    histories = {}

    batch_attempted = False

    if hasattr(
        market_provider,
        "batch_daily_history",
    ):
        batch_attempted = True

        try:
            histories = (
                market_provider.batch_daily_history(
                    needed,
                    outputsize=300,
                )
            )
        except Exception:
            histories = {}

    def get_hist(sym):
        if sym in histories:
            return histories[sym]

        if batch_attempted:
            raise RuntimeError(
                f"Dati non disponibili nel batch per {sym}"
            )

        h = market_provider.daily_history(
            sym,
            outputsize=300,
        )

        histories[sym] = h

        return h

    try:
        spy60 = base_technical(get_hist("SPY.US"))["return_60d_pct"]
    except Exception:
        spy60 = 0.0

    benchmark60 = {"SPY.US": spy60}
    for market_benchmark in market_benchmarks:
        try:
            benchmark60[market_benchmark] = base_technical(
                get_hist(market_benchmark)
            )["return_60d_pct"]
        except Exception:
            benchmark60[market_benchmark] = spy60

    sector60 = {}

    for sector in sectors:
        try:
            sector60[sector] = (
                base_technical(
                    get_hist(sector)
                )["return_60d_pct"]
            )

        except Exception:
            sector60[sector] = spy60

    quote_map = {}
    quote_batch_attempted = False
    if hasattr(market_provider, "batch_latest_quotes"):
        quote_batch_attempted = True
        try:
            quote_map = market_provider.batch_latest_quotes(symbols)
        except Exception:
            quote_map = {}

    rows = []

    for _, item in universe.iterrows():
        ticker = str(
            item["ticker"]
        ).upper()

        company = str(
            item["company"]
        )

        sector = str(
            item["sector_etf"]
        ).upper()

        benchmark = str(
            item.get("benchmark_ticker", "SPY.US")
        ).upper()

        display_ticker = str(
            item.get(
                "display_ticker",
                ticker.split(".")[0],
            )
        ).upper()

        try:
            prices = get_hist(
                ticker
            )

            if len(prices) < 65:
                raise ValueError(
                    "Storico insufficiente: "
                    "servono almeno 65 sedute."
                )

            technical = base_technical(
                prices
            )

            # Il prezzo mostrato non viene ricavato dalla serie rettificata.
            # Se la quota live/ritardata è disponibile la usiamo e ne salviamo
            # sempre fonte e orario; in caso contrario resta la chiusura raw.
            technical["last_close"] = technical["raw_eod_close"]
            technical["price_source"] = "eod_raw_close"

            if hasattr(market_provider, "latest_quote"):
                try:
                    quote = quote_map.get(ticker)
                    if quote is None and not quote_batch_attempted:
                        quote = market_provider.latest_quote(ticker)
                    if quote is None:
                        raise RuntimeError("Quota recente non disponibile nel batch.")
                    quote_price = float(quote["price"])
                    reference = float(technical["raw_eod_close"])
                    if quote_matches_reference(quote, reference):
                        technical["last_close"] = quote_price
                        technical["price_observed_at"] = quote.get("observed_at")
                        technical["price_source"] = quote.get("source")
                        technical["price_is_delayed"] = quote.get("is_delayed", True)
                        technical["previous_close"] = quote.get("previous_close")
                        technical["live_change_pct"] = quote.get("change_pct")
                        technical["price_validation"] = "ok"
                    else:
                        technical["price_validation"] = "provider_conflict"
                        technical["price_warning"] = (
                            "Quota recente incoerente con la chiusura storica; "
                            "mantenuta la chiusura verificata."
                        )
                except Exception as quote_error:
                    technical["price_validation"] = "quote_unavailable"
                    technical["price_warning"] = str(quote_error)

            metrics = {
                "revenue_growth_pct": None,
                "net_margin_pct": None,
                "liabilities_to_assets": None,
                "fcf_margin_pct": None,
                "approx_pe": None,
                "approx_ps": None,
            }

            # Il Deep Engine interroga i fondamentali solo dopo che il Light
            # Scanner ha creato la shortlist. In questo modo non scarichiamo
            # dati completi per migliaia di titoli inutilmente.
            if hasattr(market_provider, "fundamentals"):
                try:
                    metrics.update(
                        market_provider.fundamentals(ticker)
                    )
                except Exception as fundamental_error:
                    metrics["fundamentals_error"] = str(fundamental_error)

            if include_sec and sec is not None:
                sec_ticker = display_ticker

                try:
                    sec_metrics = sec.metrics(
                        sec_ticker,
                        technical["last_close"],
                    )
                    for key, value in sec_metrics.items():
                        if metrics.get(key) is None:
                            metrics[key] = value

                except Exception:
                    pass

            else:
                if (
                    "demo_revenue_growth"
                    in item.index
                ):
                    metrics.update({
                        "revenue_growth_pct": float(
                            item[
                                "demo_revenue_growth"
                            ]
                        ),
                        "net_margin_pct": float(
                            item[
                                "demo_net_margin"
                            ]
                        ),
                        "liabilities_to_assets": float(
                            item[
                                "demo_liab_assets"
                            ]
                        ),
                        "fcf_margin_pct": float(
                            item[
                                "demo_fcf_margin"
                            ]
                        ),
                    })

            # Quote/listing currency comes from the selected market symbol,
            # not from the company's reporting currency in Fundamentals.
            listing_currency = item.get("currency")
            if listing_currency is None or pd.isna(listing_currency):
                listing_currency = metrics.get("listing_currency")
            if listing_currency is None or pd.isna(listing_currency):
                exchange_code = item.get("light_exchange")
                if exchange_code is None or pd.isna(exchange_code):
                    exchange_code = ticker.rsplit(".", 1)[-1]
                listing_currency = _currency_for(exchange_code, None)
            listing_currency = str(listing_currency).upper()
            metrics["currency"] = listing_currency

            fx_rate = item.get("market_cap_fx_to_usd")
            try:
                fx_rate = float(fx_rate)
                if not math.isfinite(fx_rate) or fx_rate <= 0:
                    fx_rate = None
            except (TypeError, ValueError):
                fx_rate = None
            if fx_rate is None and hasattr(market_provider, "currency_to_usd_rate"):
                fx_rate = market_provider.currency_to_usd_rate(listing_currency)
            if fx_rate is not None:
                metrics["market_cap_fx_to_usd"] = fx_rate
                market_cap = metrics.get("market_cap")
                try:
                    if market_cap is not None and not pd.isna(market_cap):
                        metrics["market_cap_usd"] = float(market_cap) * fx_rate
                except (TypeError, ValueError):
                    pass

            metrics = reconcile_market_metrics(
                metrics,
                technical,
                provider_ticker=ticker,
                listing_currency=listing_currency,
                fx_rate=fx_rate,
            )
            metrics = enrich_fundamental_scores(metrics)
            metrics["fundamental_confidence_score"] = metrics.get("confidence_score")
            quality = metrics["quality_score"]

            trap_raw = value_trap_risk(
                metrics,
                technical,
            )
            fundamental_conf = _safe_number(metrics.get("confidence_score"))
            if (
                trap_raw is None
                or fundamental_conf is None
                or fundamental_conf < 45
                or str(metrics.get("data_validation_status") or "").lower() in {"invalid", "secondary_listing"}
            ):
                trap = None
            else:
                # Shrink extreme 0/100 readings toward the neutral prior when
                # the fundamental layer is incomplete. This avoids displaying
                # false certainty from a partial dataset.
                weight = max(0.0, min(1.0, fundamental_conf / 100.0))
                trap = 30.0 + (float(trap_raw) - 30.0) * weight
                trap = round(clamp(trap), 1)
            metrics["value_trap_risk_raw"] = trap_raw
            metrics["value_trap_confidence_score"] = fundamental_conf
            metrics["value_trap_risk"] = trap

            scores = score_one(
                technical,
                quality,
                benchmark60.get(benchmark, spy60),
                sector60.get(
                    sector,
                    benchmark60.get(benchmark, spy60),
                ),
            )
            scores["relative_60d_vs_market_pct"] = scores[
                "relative_60d_vs_spy_pct"
            ]

            recovery = recovery_potential(
                scores["anomaly_score"],
                quality,
                trap,
                scores[
                    "relative_60d_vs_sector_pct"
                ],
            )

            row = {
                "ticker": display_ticker,
                "provider_ticker": ticker,
                "company": company,
                "sector_etf": sector,
                "benchmark_ticker": benchmark,
                **technical,
                **metrics,
                **scores,
                "value_trap_risk": round(trap, 1) if trap is not None else None,
                "recovery_potential": recovery,
                "catalyst_label": (
                    "Non analizzato"
                ),
                # Un evento non analizzato resta mancante: non viene tradotto
                # in un falso valore neutro che potrebbe alterare lo score.
                "catalyst_risk": None,
                "earnings_related": False,
                "catalyst_explanation": (
                    "Catalyst engine non ancora eseguito."
                ),
                "catalyst_items": [],
                "recent_filings": [],
                "error": None,
            }

            for metadata_field in ["currency", "country"]:
                if metrics.get(metadata_field) is not None:
                    row[metadata_field] = metrics.get(metadata_field)
                elif metadata_field in item.index:
                    row[metadata_field] = item.get(metadata_field)

            for field in [
                "light_return_1d_pct",
                "light_anomaly_score",
                "light_drawdown_250d_pct",
                "light_volume_ratio",
                "light_ema50_pct",
                "light_ema200_pct",
                "light_universe_rank",
                "light_universe_scanned",
                "source_exchange",
                "light_last_price",
                "light_market_cap",
                "light_market_cap_usd",
                "market_cap_fx_to_usd",
                "light_volume",
                "light_data_date",
                "light_sector",
                "light_industry",
                "light_exchange",
                "benchmark_ticker",
                "currency",
                "country",
            ]:
                if field in item.index:
                    value = item.get(field)
                    if value is not None and not pd.isna(value):
                        row[field] = value

            row[
                "explanation"
            ] = explanation(
                row
            )

            rows.append(
                row
            )

        except Exception as error:
            rows.append({
                "ticker": display_ticker,
                "provider_ticker": ticker,
                "company": company,
                "sector_etf": sector,
                "error": str(error),
            })

    df = pd.DataFrame(
        rows
    )

    if (
        df.empty
        or "anomaly_score"
        not in df.columns
    ):
        return df

    valid_idx = (
        df[
            df["error"].isna()
        ]
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .head(
            int(catalyst_top_n)
        )
        .index
    )

    for idx in valid_idx:
        ticker = df.at[
            idx,
            "ticker",
        ]

        provider_ticker = df.at[
            idx,
            "provider_ticker",
        ]

        releases = []
        filings = []

        try:
            if hasattr(
                market_provider,
                "press_releases",
            ):
                releases = (
                    market_provider.press_releases(
                        provider_ticker,
                        limit=8,
                    )
                )
        except Exception:
            releases = []

        if include_sec and sec is not None:
            try:
                filings = (
                    sec.recent_filings(
                        ticker,
                        limit=8,
                    )
                )
            except Exception:
                filings = []

        catalyst = classify_catalysts(
            releases,
            filings,
        )

        for key, value in catalyst.items():
            df.at[
                idx,
                key,
            ] = value

    # Recompute the user-facing confidence after the catalyst pass.
    for idx in df.index:
        error = df.at[idx, "error"] if "error" in df.columns else None
        if error is not None and not pd.isna(error):
            continue
        row_dict = df.loc[idx].to_dict()
        df.at[idx, "confidence_score"] = overall_confidence(row_dict)

    opportunities = []

    for _, row in df.iterrows():
        error = row.get(
            "error"
        )

        if (
            error is not None
            and not pd.isna(error)
        ):
            opportunities.append(
                None
            )
            continue

        opportunities.append(
            opportunity_score(
                row[
                    "anomaly_score"
                ],
                row[
                    "quality_score"
                ],
                row[
                    "value_trap_risk"
                ],
                row[
                    "catalyst_risk"
                ],
                valuation_score=row.get("valuation_score"),
                financial_risk=row.get("financial_risk_score"),
                distress_risk=row.get("distress_risk_score"),
                dilution_risk=row.get("dilution_risk_score"),
                confidence_score=row.get("confidence_score", 0),
            )
        )

    df[
        "opportunity_score"
    ] = opportunities

    df = df.sort_values(
        [
            "opportunity_score",
            "anomaly_score",
        ],
        ascending=False,
        na_position="last",
    )

    return df.reset_index(
        drop=True
    )
