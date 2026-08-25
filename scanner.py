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
    exchanges=("us",),
    max_return_1d_pct=-8.0,
    min_avg_volume=200_000,
    min_price=2.0,
    min_market_cap=500_000_000,
    limit_per_exchange=100,
):
    """
    LIGHT SCANNER.

    Usa lo screener del provider per individuare
    rapidamente solo i titoli che hanno avuto
    movimenti fortemente negativi.

    Non assegna ancora il punteggio finale.
    Serve solo a creare la shortlist da passare
    all'analisi approfondita.
    """

    if not hasattr(
        market_provider,
        "screen_candidates",
    ):
        return None

    candidates = market_provider.screen_candidates(
        exchanges=exchanges,
        max_return_1d_pct=max_return_1d_pct,
        min_avg_volume=min_avg_volume,
        min_price=min_price,
        min_market_cap=min_market_cap,
        limit_per_exchange=limit_per_exchange,
    )

    if candidates is None or candidates.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "company",
                "sector_etf",
            ]
        )

    rows = []

    for _, item in candidates.iterrows():
        asset_type = str(
            item.get("type") or item.get("Type") or item.get("asset_type") or ""
        ).strip()
        if any(
            excluded in asset_type.lower()
            for excluded in ("fund", "etf", "warrant", "preferred", "bond")
        ):
            continue

        ticker = str(
            item.get("code", "")
        ).strip().upper()

        if not ticker:
            continue

        company = str(
            item.get(
                "name",
                ticker,
            )
        ).strip()

        exchange = str(
            item.get(
                "exchange",
                "US",
            )
        ).strip().upper()

        api_ticker = (
            ticker
            if "." in ticker
            else f"{ticker}.{exchange}"
        )

        sector_name = str(item.get("sector") or "").strip()
        sector_etf = _benchmark_for(exchange, sector_name)

        rows.append({
            "ticker": api_ticker,
            "display_ticker": ticker,
            "company": company,
            "sector_etf": sector_etf,
            "benchmark_ticker": _market_benchmark(exchange),
            "light_return_1d_pct": item.get(
                "refund_1d_p"
            ),
            "light_last_price": item.get(
                "adjusted_close"
            ),
            "light_market_cap": item.get(
                "market_capitalization"
            ),
            "light_volume": item.get(
                "avgvol_200d"
            ),
            "light_data_date": item.get("last_day_data_date"),
            "light_sector": item.get(
                "sector"
            ),
            "light_industry": item.get(
                "industry"
            ),
            "light_exchange": exchange,
            "currency": _currency_for(
                exchange,
                item.get("currency") or item.get("currency_symbol"),
            ),
            "country": item.get("country") or item.get("country_name"),
            "asset_type": asset_type or "Common Stock",
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
    if code in {"US", "NYSE", "NASDAQ", "AMEX"}:
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
        "US": "USD", "NYSE": "USD", "NASDAQ": "USD", "AMEX": "USD",
        "LSE": "GBP", "TO": "CAD", "V": "CAD", "PA": "EUR",
        "XETRA": "EUR", "MI": "EUR", "AS": "EUR", "BR": "EUR",
        "MC": "EUR", "LS": "EUR", "SW": "CHF", "ST": "SEK",
        "CO": "DKK", "HE": "EUR", "OL": "NOK", "TSE": "JPY",
        "HK": "HKD", "AU": "AUD", "AX": "AUD", "JSE": "ZAR",
    }
    return mapping.get(code, "USD")


def _benchmark_for(exchange: str, sector: str) -> str:
    code = str(exchange or "US").upper()
    if code in {"US", "NYSE", "NASDAQ", "AMEX"}:
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

            listing_currency = metrics.get("currency")
            if listing_currency is None or pd.isna(listing_currency):
                listing_currency = item.get("currency")
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

            metrics = enrich_fundamental_scores(metrics)
            quality = metrics["quality_score"]

            trap = value_trap_risk(
                metrics,
                technical,
            )

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
