from catalyst_engine import opportunity_score
from fundamentals import enrich_fundamental_scores, value_trap_risk
from providers.eodhd import EODHDProvider
from model import live_score


def _eodhd_fixture():
    return {
        "General": {"Name": "Example Software", "Sector": "Technology"},
        "Highlights": {
            "MarketCapitalization": 70_000_000_000,
            "EBITDA": 2_000_000_000,
            "PERatio": 70,
            "PEGRatio": 4.5,
            "RevenueTTM": 10_000_000_000,
            "GrossProfitTTM": 7_000_000_000,
            "QuarterlyRevenueGrowthYOY": 0.08,
            "QuarterlyEarningsGrowthYOY": -0.10,
            "OperatingMarginTTM": 0.12,
            "ProfitMargin": 0.08,
            "ReturnOnEquityTTM": 0.14,
        },
        "Valuation": {
            "TrailingPE": 70,
            "ForwardPE": 58,
            "EnterpriseValueRevenue": 8.5,
            "EnterpriseValueEbitda": 35,
            "PriceBookMRQ": 12,
        },
        "SharesStats": {"SharesOutstanding": 1_050_000_000},
        "Financials": {
            "Income_Statement": {
                "quarterly": {
                    "2026-06-30": {
                        "date": "2026-06-30",
                        "totalRevenue": 2_500_000_000,
                        "operatingIncome": 300_000_000,
                        "netIncome": 200_000_000,
                        "interestExpense": -25_000_000,
                    }
                }
            },
            "Balance_Sheet": {
                "quarterly": {
                    "2026-06-30": {
                        "date": "2026-06-30",
                        "totalAssets": 20_000_000_000,
                        "totalLiab": 12_000_000_000,
                        "totalCurrentAssets": 5_000_000_000,
                        "totalCurrentLiabilities": 4_000_000_000,
                        "shortLongTermDebtTotal": 6_000_000_000,
                        "cashAndShortTermInvestments": 2_000_000_000,
                    }
                },
                "yearly": {
                    "2025-12-31": {
                        "date": "2025-12-31",
                        "commonStockSharesOutstanding": 1_050_000_000,
                    },
                    "2024-12-31": {
                        "date": "2024-12-31",
                        "commonStockSharesOutstanding": 1_000_000_000,
                    },
                },
            },
            "Cash_Flow": {
                "quarterly": {
                    "2026-06-30": {
                        "date": "2026-06-30",
                        "totalCashFromOperatingActivities": 400_000_000,
                        "capitalExpenditures": -100_000_000,
                    }
                }
            },
        },
    }


def test_eodhd_fundamentals_feed_all_risk_scores():
    metrics = EODHDProvider._parse_fundamentals(_eodhd_fixture(), "EXM.US")
    scored = enrich_fundamental_scores(metrics)
    trap = value_trap_risk(scored, {"return_60d_pct": -30})
    cheaper = dict(scored, valuation_score=80)
    cheaper_trap = value_trap_risk(cheaper, {"return_60d_pct": -30})

    assert metrics["revenue_growth_pct"] == 8.0
    assert metrics["pe_ratio"] == 70.0
    assert metrics["shares_outstanding_growth_pct"] == 5.0
    assert scored["valuation_score"] < 40
    assert scored["confidence_score"] >= 80
    assert trap > cheaper_trap


def test_expensive_fragile_incomplete_company_is_penalized():
    common = dict(
        anomaly_score=90,
        quality_score=65,
        value_trap_risk=45,
        catalyst_risk=45,
    )
    sound = opportunity_score(
        **common,
        valuation_score=80,
        financial_risk=20,
        distress_risk=20,
        dilution_risk=15,
        confidence_score=100,
    )
    expensive = opportunity_score(
        **common,
        valuation_score=15,
        financial_risk=75,
        distress_risk=80,
        dilution_risk=70,
        confidence_score=35,
    )

    assert expensive < sound - 25


def test_missing_quality_is_excluded_instead_of_becoming_neutral_or_zero():
    components = {
        "score_drawdown": 40,
        "score_rsi": 40,
        "score_volume": 40,
        "score_momentum": 40,
        "score_shock": 40,
        "score_market_relative": 40,
        "score_sector_relative": 40,
    }

    assert live_score(components, None) == 40.0
