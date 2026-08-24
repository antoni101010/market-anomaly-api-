from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


class EODHDProvider:
    """Provider dati di mercato EODHD per Market Anomaly."""

    BASE = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "data/price_cache",
        timeout: int = 30,
    ):
        if not api_key:
            raise ValueError("EODHD_API_KEY mancante")

        self.api_key = api_key
        self.timeout = int(timeout)

        self.cache_dir = Path(cache_dir) / "eodhd"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.cache: dict[tuple[str, int], pd.DataFrame] = {}

    @staticmethod
    def _api_symbol(symbol: str, exchange: str = "US") -> str:
        symbol = str(symbol).strip().upper()

        if "." in symbol:
            return symbol

        return f"{symbol}.{exchange.upper()}"

    def _request(self, path: str, params: dict | None = None):
        query = dict(params or {})

        query["api_token"] = self.api_key
        query.setdefault("fmt", "json")

        response = self.session.get(
            f"{self.BASE}/{path.lstrip('/')}",
            params=query,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))

        return data

    def _cache_path(self, symbol: str, outputsize: int) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(".", "_")

        return self.cache_dir / (
            f"{safe_symbol}_{int(outputsize)}.csv"
        )

    def _fundamentals_cache_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(".", "_")

        return self.cache_dir / (
            f"{safe_symbol}_fundamentals.json"
        )

    @staticmethod
    def _number(value):
        try:
            number = float(value)

            if pd.isna(number):
                return None

            return number

        except (TypeError, ValueError):
            return None

    @classmethod
    def _percent(cls, value):
        number = cls._number(value)

        if number is None:
            return None

        # EODHD restituisce normalmente
        # margini e crescite come frazioni.
        return (
            number * 100.0
            if abs(number) <= 2.0
            else number
        )

    @staticmethod
    def _latest_periods(
        section: dict | None,
        limit: int = 4,
    ) -> list[dict]:

        if not isinstance(section, dict):
            return []

        values = [
            value
            for value in section.values()
            if isinstance(value, dict)
        ]

        return sorted(
            values,
            key=lambda value: str(
                value.get("date")
                or value.get("filing_date")
                or ""
            ),
            reverse=True,
        )[:limit]

    @classmethod
    def _sum_periods(
        cls,
        periods: list[dict],
        *keys: str,
    ):
        values = []

        for period in periods:
            value = next(
                (
                    cls._number(period.get(key))
                    for key in keys
                    if cls._number(
                        period.get(key)
                    ) is not None
                ),
                None,
            )

            if value is not None:
                values.append(value)

        return (
            sum(values)
            if values
            else None
        )

    @classmethod
    def _parse_fundamentals(
        cls,
        data: dict,
        symbol: str,
    ) -> dict:

        if not isinstance(data, dict) or not data:
            raise RuntimeError(
                f"Fondamentali EODHD non disponibili per {symbol}"
            )

        general = data.get("General") or {}
        highlights = data.get("Highlights") or {}
        valuation = data.get("Valuation") or {}
        shares = data.get("SharesStats") or {}
        financials = data.get("Financials") or {}

        income = (
            financials.get("Income_Statement")
            or {}
        )

        balance = (
            financials.get("Balance_Sheet")
            or {}
        )

        cash_flow = (
            financials.get("Cash_Flow")
            or {}
        )

        income_q = cls._latest_periods(
            income.get("quarterly"),
            4,
        )

        balance_q = cls._latest_periods(
            balance.get("quarterly"),
            2,
        )

        cash_q = cls._latest_periods(
            cash_flow.get("quarterly"),
            4,
        )

        balance_y = cls._latest_periods(
            balance.get("yearly"),
            2,
        )

        latest_income = (
            income_q[0]
            if income_q
            else {}
        )

        latest_balance = (
            balance_q[0]
            if balance_q
            else (
                balance_y[0]
                if balance_y
                else {}
            )
        )

        revenue_ttm = cls._number(
            highlights.get("RevenueTTM")
        )

        if revenue_ttm is None:
            revenue_ttm = cls._sum_periods(
                income_q,
                "totalRevenue",
                "revenue",
            )

        gross_profit_ttm = cls._number(
            highlights.get("GrossProfitTTM")
        )

        if gross_profit_ttm is None:
            gross_profit_ttm = cls._sum_periods(
                income_q,
                "grossProfit",
            )

        operating_income_ttm = cls._sum_periods(
            income_q,
            "operatingIncome",
        )

        net_income_ttm = cls._sum_periods(
            income_q,
            "netIncome",
        )

        operating_cash_ttm = cls._sum_periods(
            cash_q,
            "totalCashFromOperatingActivities",
            "cashFromOperatingActivities",
        )

        capex_ttm = cls._sum_periods(
            cash_q,
            "capitalExpenditures",
            "capitalExpenditure",
        )

        free_cash_flow = cls._sum_periods(
            cash_q,
            "freeCashFlow",
        )

        if (
            free_cash_flow is None
            and operating_cash_ttm is not None
            and capex_ttm is not None
        ):
            free_cash_flow = (
                operating_cash_ttm + capex_ttm
                if capex_ttm < 0
                else operating_cash_ttm - capex_ttm
            )

        market_cap = cls._number(
            highlights.get(
                "MarketCapitalization"
            )
        )

        ebitda = cls._number(
            highlights.get("EBITDA")
        )

        if ebitda is None:
            ebitda = cls._sum_periods(
                income_q,
                "ebitda",
            )

        total_assets = cls._number(
            latest_balance.get("totalAssets")
        )

        total_liabilities = cls._number(
            latest_balance.get("totalLiab")
            or latest_balance.get(
                "totalLiabilities"
            )
        )

        current_assets = cls._number(
            latest_balance.get(
                "totalCurrentAssets"
            )
        )

        current_liabilities = cls._number(
            latest_balance.get(
                "totalCurrentLiabilities"
            )
        )

        total_debt = cls._number(
            latest_balance.get(
                "shortLongTermDebtTotal"
            )
            or latest_balance.get("totalDebt")
            or latest_balance.get(
                "shortLongTermDebt"
            )
        )

        cash = cls._number(
            latest_balance.get(
                "cashAndShortTermInvestments"
            )
            or latest_balance.get("cash")
            or latest_balance.get(
                "cashAndCashEquivalents"
            )
        )

        interest_expense = cls._sum_periods(
            income_q,
            "interestExpense",
        )

        if interest_expense is not None:
            interest_expense = abs(
                interest_expense
            )

        # Per la crescita delle azioni usiamo
        # due periodi annuali omogenei.
        current_shares = (
            cls._number(
                balance_y[0].get(
                    "commonStockSharesOutstanding"
                )
            )
            if balance_y
            else cls._number(
                shares.get("SharesOutstanding")
                or latest_balance.get(
                    "commonStockSharesOutstanding"
                )
            )
        )

        previous_shares = None

        if len(balance_y) >= 2:
            previous_shares = cls._number(
                balance_y[1].get(
                    "commonStockSharesOutstanding"
                )
            )

        def ratio(
            numerator,
            denominator,
            multiplier=1.0,
        ):
            if (
                numerator is None
                or denominator in (None, 0)
            ):
                return None

            return (
                numerator
                / denominator
                * multiplier
            )

        fcf_margin = ratio(
            free_cash_flow,
            revenue_ttm,
            100.0,
        )

        negative_fcf = (
            free_cash_flow is not None
            and free_cash_flow < 0
        )

        cash_runway = (
            ratio(
                cash,
                abs(free_cash_flow),
                12.0,
            )
            if negative_fcf
            else None
        )

        return {
            "fundamentals_source": "eodhd",
            "fundamentals_symbol": symbol,
            "company_name": general.get("Name"),
            "sector": general.get("Sector"),
            "industry": general.get("Industry"),
            "market_cap": market_cap,
            "revenue_ttm": revenue_ttm,
            "free_cash_flow_ttm": free_cash_flow,
            "revenue_growth_pct": cls._percent(
                highlights.get(
                    "QuarterlyRevenueGrowthYOY"
                )
            ),
            "eps_growth_pct": cls._percent(
                highlights.get(
                    "QuarterlyEarningsGrowthYOY"
                )
            ),
            "gross_margin_pct": ratio(
                gross_profit_ttm,
                revenue_ttm,
                100.0,
            ),
            "operating_margin_pct": (
                cls._percent(
                    highlights.get(
                        "OperatingMarginTTM"
                    )
                )
                or ratio(
                    operating_income_ttm,
                    revenue_ttm,
                    100.0,
                )
            ),
            "net_margin_pct": (
                cls._percent(
                    highlights.get(
                        "ProfitMargin"
                    )
                )
                or ratio(
                    net_income_ttm,
                    revenue_ttm,
                    100.0,
                )
            ),
            "fcf_margin_pct": fcf_margin,
            "roe_pct": cls._percent(
                highlights.get(
                    "ReturnOnEquityTTM"
                )
            ),
            "pe_ratio": (
                cls._number(
                    highlights.get("PERatio")
                )
                or cls._number(
                    valuation.get("TrailingPE")
                )
            ),
            "forward_pe": cls._number(
                valuation.get("ForwardPE")
            ),
            "ev_to_ebitda": cls._number(
                valuation.get(
                    "EnterpriseValueEbitda"
                )
            ),
            "ev_to_sales": cls._number(
                valuation.get(
                    "EnterpriseValueRevenue"
                )
            ),
            "price_to_book": cls._number(
                valuation.get("PriceBookMRQ")
            ),
            "peg_ratio": cls._number(
                highlights.get("PEGRatio")
            ),
            "fcf_yield_pct": ratio(
                free_cash_flow,
                market_cap,
                100.0,
            ),
            "debt_to_ebitda": ratio(
                total_debt,
                ebitda,
            ),
            "net_debt_to_ebitda": ratio(
                (
                    total_debt - (cash or 0.0)
                    if total_debt is not None
                    else None
                ),
                ebitda,
            ),
            "liabilities_to_assets": ratio(
                total_liabilities,
                total_assets,
            ),
            "current_ratio": ratio(
                current_assets,
                current_liabilities,
            ),
            "interest_coverage": ratio(
                (
                    operating_income_ttm
                    or cls._number(
                        latest_income.get(
                            "operatingIncome"
                        )
                    )
                ),
                interest_expense,
            ),
            "cash_runway_months": cash_runway,
            "shares_outstanding_growth_pct": (
                ratio(
                    current_shares
                    - previous_shares,
                    previous_shares,
                    100.0,
                )
                if (
                    current_shares is not None
                    and previous_shares is not None
                )
                else None
            ),
        }

    def fundamentals(
        self,
        symbol: str,
        max_age_hours: int = 24,
    ) -> dict:

        api_symbol = self._api_symbol(
            symbol
        )

        cache_path = (
            self._fundamentals_cache_path(
                api_symbol
            )
        )

        if cache_path.exists():
            age_seconds = (
                time.time()
                - cache_path.stat().st_mtime
            )

            if (
                age_seconds
                <= max(
                    1,
                    int(max_age_hours),
                )
                * 3600
            ):
                try:
                    return self._parse_fundamentals(
                        json.loads(
                            cache_path.read_text(
                                encoding="utf-8"
                            )
                        ),
                        api_symbol,
                    )

                except (
                    OSError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                ):
                    pass

        data = self._request(
            f"fundamentals/{api_symbol}"
        )

        cache_path.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

        return self._parse_fundamentals(
            data,
            api_symbol,
        )

    @staticmethod
    def _parse_history(
        data,
        symbol: str,
        outputsize: int,
    ) -> pd.DataFrame:

        if not isinstance(data, list) or not data:
            raise RuntimeError(
                f"Nessun dato EODHD per {symbol}"
            )

        df = pd.DataFrame(data)

        if (
            "date" not in df.columns
            or "close" not in df.columns
        ):
            raise RuntimeError(
                f"Risposta EODHD non valida per {symbol}"
            )

        if "volume" not in df.columns:
            df["volume"] = 0

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            if column not in df.columns:
                if column == "volume":
                    df[column] = 0
                else:
                    raise RuntimeError(
                        f"Campo {column} mancante per {symbol}"
                    )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df["datetime"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = (
            df.dropna(
                subset=["datetime", "close"]
            )
            .sort_values("datetime")
            .tail(int(outputsize))[
                [
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            .reset_index(drop=True)
        )

        if df.empty:
            raise RuntimeError(
                f"Storico EODHD vuoto per {symbol}"
            )

        return df

    def daily_history(
        self,
        symbol: str,
        outputsize: int = 300,
        adjust: str = "splits",
    ) -> pd.DataFrame:

        key = (
            str(symbol).upper(),
            int(outputsize),
        )

        if key in self.cache:
            return self.cache[key].copy()

        path = self._cache_path(*key)

        if path.exists():
            try:
                cached = pd.read_csv(
                    path,
                    parse_dates=["datetime"],
                )

                if len(cached) >= min(
                    65,
                    int(outputsize),
                ):
                    self.cache[key] = cached
                    return cached.copy()

            except Exception:
                pass

        api_symbol = self._api_symbol(
            symbol
        )

        data = self._request(
            f"eod/{api_symbol}",
            params={
                "period": "d",
                "order": "a",
            },
        )

        df = self._parse_history(
            data,
            api_symbol,
            outputsize,
        )

        df.to_csv(
            path,
            index=False,
        )

        self.cache[key] = df

        return df.copy()

    def screen_candidates(
        self,
        exchanges: Iterable[str] = ("us",),
        max_return_1d_pct: float = -8.0,
        min_avg_volume: int = 200_000,
        min_price: float = 2.0,
        min_market_cap: float | None = 500_000_000,
        limit_per_exchange: int = 100,
    ) -> pd.DataFrame:
        """
        Prima scansione veloce.

        Cerca titoli con forti movimenti negativi.
        Non fornisce consigli di acquisto o vendita.
        """

        rows: list[dict] = []

        for exchange in exchanges:
            filters = [
                [
                    "exchange",
                    "=",
                    str(exchange).lower(),
                ],
                [
                    "refund_1d_p",
                    "<=",
                    float(max_return_1d_pct),
                ],
                [
                    "avgvol_1d",
                    ">=",
                    int(min_avg_volume),
                ],
                [
                    "adjusted_close",
                    ">=",
                    float(min_price),
                ],
            ]

            if min_market_cap is not None:
                filters.append(
                    [
                        "market_capitalization",
                        ">=",
                        float(min_market_cap),
                    ]
                )

            data = self._request(
                "screener",
                params={
                    "filters": json.dumps(
                        filters,
                        separators=(",", ":"),
                    ),
                    "sort": "refund_1d_p.asc",
                    "limit": max(
                        1,
                        min(
                            int(limit_per_exchange),
                            500,
                        ),
                    ),
                    "offset": 0,
                },
            )

            if isinstance(data, dict):
                data = data.get(
                    "data",
                    [],
                )

            if isinstance(data, list):
                rows.extend(data)

        return pd.DataFrame(rows)
