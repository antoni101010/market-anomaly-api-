from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
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
        daily_cache_ttl_minutes: int = 30,
        live_quote_ttl_seconds: int = 60,
        retry_count: int = 2,
        screener_max_requests: int = 25,
        bulk_cache_ttl_minutes: int = 45,
    ):
        if not api_key:
            raise ValueError("EODHD_API_KEY mancante")

        self.api_key = api_key
        self.timeout = int(timeout)
        self.daily_cache_ttl_seconds = max(
            60,
            int(daily_cache_ttl_minutes) * 60,
        )
        self.live_quote_ttl_seconds = max(
            5,
            int(live_quote_ttl_seconds),
        )
        self.retry_count = max(0, min(int(retry_count), 5))
        self.screener_max_requests = max(1, int(screener_max_requests))
        self.bulk_cache_ttl_seconds = max(300, int(bulk_cache_ttl_minutes) * 60)

        self.cache_dir = Path(cache_dir) / "eodhd"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.cache: dict[tuple[str, int], pd.DataFrame] = {}
        self.cache_times: dict[tuple[str, int], float] = {}
        self.quote_cache: dict[str, tuple[float, dict]] = {}
        self.last_screener_stats: dict = {}
        self.last_bulk_stats: dict = {}
        self._fundamentals_denied_at = 0.0
        self._quotes_denied_at = 0.0
        self._intraday_denied_at = 0.0
        self.capability_retry_seconds = 300

    def _capability_blocked(self, denied_at: float) -> bool:
        return bool(
            denied_at
            and time.time() - denied_at < self.capability_retry_seconds
        )

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

        response = None
        for attempt in range(self.retry_count + 1):
            response = self.session.get(
                f"{self.BASE}/{path.lstrip('/')}",
                params=query,
                timeout=self.timeout,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt < self.retry_count:
                time.sleep(0.6 * (2 ** attempt))

        if response is None:
            raise RuntimeError("EODHD: nessuna risposta dal provider.")

        if response.status_code >= 400:
            if response.status_code == 403:
                raise RuntimeError(
                    "EODHD: questo endpoint non è incluso "
                    "nel piano attuale (HTTP 403)."
                )

            raise RuntimeError(
                "EODHD: richiesta non riuscita "
                f"(HTTP {response.status_code})."
            )

        try:
            data = response.json()
        except ValueError as error:
            raise RuntimeError(
                "EODHD: risposta non valida dal provider."
            ) from error

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
        return self.cache_dir / f"{safe_symbol}_fundamentals.json"

    def _exchange_symbols_cache_path(self, exchange: str) -> Path:
        safe_exchange = str(exchange).strip().upper().replace("/", "_")
        return self.cache_dir / f"exchange_symbols_{safe_exchange}.json"

    def _bulk_eod_cache_path(self, exchange: str) -> Path:
        safe_exchange = str(exchange).strip().upper().replace("/", "_")
        return self.cache_dir / f"bulk_eod_extended_{safe_exchange}.json"

    @staticmethod
    def _age_seconds(path: Path) -> float:
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            return float("inf")

    @staticmethod
    def _number(value):
        try:
            number = float(value)
            if pd.isna(number):
                return None
            return number
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_currency(value) -> str:
        raw = str(value or "").strip().upper()
        aliases = {
            "$": "USD", "US$": "USD", "USD": "USD",
            "€": "EUR", "EUR": "EUR",
            "£": "GBP", "GBP": "GBP",
            "GBX": "GBX", "GBPENCE": "GBX", "GBP PENCE": "GBX", "GBPENCE": "GBX",
            "P": "GBX", "GBp".upper(): "GBX",
            "¥": "JPY", "JPY": "JPY",
            "C$": "CAD", "CAD": "CAD",
            "A$": "AUD", "AUD": "AUD",
            "HK$": "HKD", "HKD": "HKD",
            "CHF": "CHF", "SEK": "SEK", "NOK": "NOK", "DKK": "DKK",
            "ZAR": "ZAR", "PLN": "PLN", "CNY": "CNY",
        }
        return aliases.get(raw, raw if len(raw) == 3 and raw.isalpha() else "")

    @staticmethod
    def _major_currency_factor(currency: str) -> float:
        # London securities quoted in GBX are expressed in pence. Monetary
        # aggregates such as market cap must be converted to pounds before FX.
        return 0.01 if str(currency or "").upper() == "GBX" else 1.0

    @classmethod
    def _percent(cls, value):
        number = cls._number(value)
        if number is None:
            return None
        # EODHD restituisce normalmente margini e crescite come frazioni.
        return number * 100.0 if abs(number) <= 2.0 else number

    @staticmethod
    def _latest_periods(section: dict | None, limit: int = 4) -> list[dict]:
        if not isinstance(section, dict):
            return []
        values = [value for value in section.values() if isinstance(value, dict)]
        return sorted(
            values,
            key=lambda value: str(value.get("date") or value.get("filing_date") or ""),
            reverse=True,
        )[:limit]

    @classmethod
    def _sum_periods(cls, periods: list[dict], *keys: str):
        values = []
        for period in periods:
            value = next((cls._number(period.get(key)) for key in keys if cls._number(period.get(key)) is not None), None)
            if value is not None:
                values.append(value)
        return sum(values) if values else None

    @classmethod
    def _parse_fundamentals(cls, data: dict, symbol: str) -> dict:
        if not isinstance(data, dict) or not data:
            raise RuntimeError(f"Fondamentali EODHD non disponibili per {symbol}")

        general = data.get("General") or {}
        highlights = data.get("Highlights") or {}
        valuation = data.get("Valuation") or {}
        shares = data.get("SharesStats") or {}
        financials = data.get("Financials") or {}

        income = financials.get("Income_Statement") or {}
        balance = financials.get("Balance_Sheet") or {}
        cash_flow = financials.get("Cash_Flow") or {}

        income_q = cls._latest_periods(income.get("quarterly"), 4)
        balance_q = cls._latest_periods(balance.get("quarterly"), 2)
        cash_q = cls._latest_periods(cash_flow.get("quarterly"), 4)
        balance_y = cls._latest_periods(balance.get("yearly"), 2)

        latest_income = income_q[0] if income_q else {}
        latest_balance = balance_q[0] if balance_q else (balance_y[0] if balance_y else {})

        revenue_ttm = cls._number(highlights.get("RevenueTTM"))
        if revenue_ttm is None:
            revenue_ttm = cls._sum_periods(income_q, "totalRevenue", "revenue")

        gross_profit_ttm = cls._number(highlights.get("GrossProfitTTM"))
        if gross_profit_ttm is None:
            gross_profit_ttm = cls._sum_periods(income_q, "grossProfit")

        operating_income_ttm = cls._sum_periods(income_q, "operatingIncome")
        net_income_ttm = cls._sum_periods(income_q, "netIncome")
        operating_cash_ttm = cls._sum_periods(
            cash_q, "totalCashFromOperatingActivities", "cashFromOperatingActivities"
        )
        capex_ttm = cls._sum_periods(cash_q, "capitalExpenditures", "capitalExpenditure")
        free_cash_flow = cls._sum_periods(cash_q, "freeCashFlow")
        if free_cash_flow is None and operating_cash_ttm is not None and capex_ttm is not None:
            free_cash_flow = operating_cash_ttm + capex_ttm if capex_ttm < 0 else operating_cash_ttm - capex_ttm

        market_cap = cls._number(highlights.get("MarketCapitalization"))
        ebitda = cls._number(highlights.get("EBITDA"))
        if ebitda is None:
            ebitda = cls._sum_periods(income_q, "ebitda")

        total_assets = cls._number(latest_balance.get("totalAssets"))
        total_liabilities = cls._number(
            latest_balance.get("totalLiab") or latest_balance.get("totalLiabilities")
        )
        current_assets = cls._number(latest_balance.get("totalCurrentAssets"))
        current_liabilities = cls._number(latest_balance.get("totalCurrentLiabilities"))
        total_debt = cls._number(
            latest_balance.get("shortLongTermDebtTotal")
            or latest_balance.get("totalDebt")
            or latest_balance.get("shortLongTermDebt")
        )
        cash = cls._number(
            latest_balance.get("cashAndShortTermInvestments")
            or latest_balance.get("cash")
            or latest_balance.get("cashAndCashEquivalents")
        )

        interest_expense = cls._sum_periods(income_q, "interestExpense")
        if interest_expense is not None:
            interest_expense = abs(interest_expense)

        # Per la crescita usiamo due periodi annuali omogenei. Il dato corrente
        # di SharesStats resta il fallback quando lo storico non è disponibile.
        current_shares = (
            cls._number(balance_y[0].get("commonStockSharesOutstanding"))
            if balance_y
            else cls._number(
                shares.get("SharesOutstanding")
                or latest_balance.get("commonStockSharesOutstanding")
            )
        )
        previous_shares = None
        if len(balance_y) >= 2:
            previous_shares = cls._number(balance_y[1].get("commonStockSharesOutstanding"))

        def ratio(numerator, denominator, multiplier=1.0):
            if numerator is None or denominator in (None, 0):
                return None
            return numerator / denominator * multiplier

        fcf_margin = ratio(free_cash_flow, revenue_ttm, 100.0)
        negative_fcf = free_cash_flow is not None and free_cash_flow < 0
        cash_runway = (
            ratio(cash, abs(free_cash_flow), 12.0)
            if negative_fcf
            else None
        )
        cash_runway_status = (
            "calculated_negative_fcf" if negative_fcf
            else "not_applicable_positive_fcf" if free_cash_flow is not None
            else "unavailable"
        )

        price_to_sales = (
            cls._number(valuation.get("PriceSalesTTM"))
            or cls._number(valuation.get("PriceSalesMRQ"))
            or ratio(market_cap, revenue_ttm)
        )
        price_to_sales_source = (
            "provider_multiple"
            if cls._number(valuation.get("PriceSalesTTM")) is not None
            or cls._number(valuation.get("PriceSalesMRQ")) is not None
            else "market_cap_div_revenue_ttm" if price_to_sales is not None
            else None
        )

        period_end = latest_income.get("date") or latest_balance.get("date")
        fundamentals_age_days = None
        if period_end:
            try:
                period_ts = pd.Timestamp(period_end)
                if period_ts.tzinfo is None:
                    period_ts = period_ts.tz_localize("UTC")
                fundamentals_age_days = max(0, int((pd.Timestamp.now(tz="UTC") - period_ts).days))
            except Exception:
                fundamentals_age_days = None

        # EODHD documents General.CurrencyCode as the currency of the listing.
        # Keep that separate from the selected quote metadata so cross-listed
        # securities cannot silently inherit USD formatting or ratios.
        fundamentals_currency = cls._normalize_currency(
            general.get("CurrencyCode") or general.get("CurrencyName")
        )
        primary_ticker = str(general.get("PrimaryTicker") or "").strip().upper()
        shares_outstanding = cls._number(
            shares.get("SharesOutstanding")
            or latest_balance.get("commonStockSharesOutstanding")
        )
        eps_ttm = cls._number(
            highlights.get("EarningsShare")
            or highlights.get("DilutedEpsTTM")
        )
        provider_pe = (
            cls._number(highlights.get("PERatio"))
            or cls._number(valuation.get("TrailingPE"))
        )

        return {
            "fundamentals_source": "eodhd",
            "fundamentals_symbol": symbol,
            "fundamentals_period_end": period_end,
            "fundamentals_age_days": fundamentals_age_days,
            # CurrencyCode is retained as the company's/reporting currency.
            # The actual listing/quote currency is resolved by the scanner from
            # the selected exchange symbol and must not be overwritten here.
            "fundamentals_currency": fundamentals_currency,
            # Backward-compatible alias for the current mobile schema.
            "reporting_currency": fundamentals_currency,
            "fundamentals_exchange": general.get("Exchange"),
            "primary_ticker": primary_ticker,
            "fundamentals_isin": general.get("ISIN"),
            "fundamentals_updated_at": general.get("UpdatedAt"),
            "fundamentals_type": general.get("Type"),
            "fundamentals_is_delisted": bool(general.get("IsDelisted") or False),
            "shares_outstanding": shares_outstanding,
            "eps_ttm": eps_ttm,
            "exchange": general.get("Exchange"),
            "country": general.get("CountryName") or general.get("CountryISO"),
            "company_name": general.get("Name"),
            "sector": general.get("Sector"),
            "industry": general.get("Industry"),
            "market_cap": market_cap,
            "provider_market_cap": market_cap,
            "market_cap_usd": market_cap
            if fundamentals_currency == "USD"
            else None,
            "revenue_ttm": revenue_ttm,
            "free_cash_flow_ttm": free_cash_flow,
            "revenue_growth_pct": cls._percent(highlights.get("QuarterlyRevenueGrowthYOY")),
            "eps_growth_pct": cls._percent(highlights.get("QuarterlyEarningsGrowthYOY")),
            "gross_margin_pct": ratio(gross_profit_ttm, revenue_ttm, 100.0),
            "operating_margin_pct": cls._percent(highlights.get("OperatingMarginTTM"))
            or ratio(operating_income_ttm, revenue_ttm, 100.0),
            "net_margin_pct": cls._percent(highlights.get("ProfitMargin"))
            or ratio(net_income_ttm, revenue_ttm, 100.0),
            "fcf_margin_pct": fcf_margin,
            "roe_pct": cls._percent(highlights.get("ReturnOnEquityTTM")),
            "pe_ratio": provider_pe,
            "provider_pe_ratio": provider_pe,
            "forward_pe": cls._number(valuation.get("ForwardPE")),
            "ev_to_ebitda": cls._number(valuation.get("EnterpriseValueEbitda")),
            "ev_to_sales": cls._number(valuation.get("EnterpriseValueRevenue")),
            "price_to_sales": price_to_sales,
            "price_to_sales_source": price_to_sales_source,
            "price_to_book": cls._number(valuation.get("PriceBookMRQ")),
            "peg_ratio": cls._number(highlights.get("PEGRatio")),
            "fcf_yield_pct": ratio(free_cash_flow, market_cap, 100.0),
            "debt_to_ebitda": ratio(total_debt, ebitda),
            "net_debt_to_ebitda": ratio(
                total_debt - (cash or 0.0) if total_debt is not None else None,
                ebitda,
            ),
            "liabilities_to_assets": ratio(total_liabilities, total_assets),
            "current_ratio": ratio(current_assets, current_liabilities),
            "interest_coverage": ratio(
                operating_income_ttm or cls._number(latest_income.get("operatingIncome")),
                interest_expense,
            ),
            "cash_runway_months": cash_runway,
            "cash_runway_status": cash_runway_status,
            "pe_basis": "TTM/trailing",
            "shares_outstanding_growth_pct": (
                ratio(current_shares - previous_shares, previous_shares, 100.0)
                if current_shares is not None and previous_shares is not None
                else None
            ),
        }

    def fundamentals(self, symbol: str, max_age_hours: int = 24) -> dict:
        if self._capability_blocked(self._fundamentals_denied_at):
            raise RuntimeError(
                "EODHD: Fundamentals non incluso nel piano attuale (HTTP 403)."
            )
        api_symbol = self._api_symbol(symbol)
        cache_path = self._fundamentals_cache_path(api_symbol)

        if cache_path.exists():
            age_seconds = time.time() - cache_path.stat().st_mtime
            if age_seconds <= max(1, int(max_age_hours)) * 3600:
                try:
                    return self._parse_fundamentals(
                        json.loads(cache_path.read_text(encoding="utf-8")),
                        api_symbol,
                    )
                except (OSError, ValueError, TypeError, RuntimeError):
                    pass

        try:
            data = self._request(f"v1.1/fundamentals/{api_symbol}")
        except RuntimeError as error:
            if "HTTP 403" in str(error):
                self._fundamentals_denied_at = time.time()
            raise
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return self._parse_fundamentals(data, api_symbol)

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

        if "date" not in df.columns or "close" not in df.columns:
            raise RuntimeError(
                f"Risposta EODHD non valida per {symbol}"
            )

        if "volume" not in df.columns:
            df["volume"] = 0

        # Conserviamo sia il prezzo realmente battuto sia la serie rettificata.
        # I calcoli storici usano la serie rettificata per evitare falsi crolli
        # causati da split; il prezzo mostrato viene poi sostituito dalla quota
        # più recente non rettificata.
        df["raw_close"] = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        if "adjusted_close" in df.columns:
            df["adjusted_close"] = pd.to_numeric(
                df["adjusted_close"],
                errors="coerce",
            )
        else:
            df["adjusted_close"] = df["raw_close"]

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

        factor = (
            df["adjusted_close"] / df["raw_close"]
        ).replace([float("inf"), float("-inf")], pd.NA)
        factor = factor.fillna(1.0)

        for column in ["open", "high", "low"]:
            df[f"raw_{column}"] = df[column]
            df[column] = df[column] * factor

        df["close"] = df["adjusted_close"].fillna(df["raw_close"])

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
                    "raw_open",
                    "raw_high",
                    "raw_low",
                    "raw_close",
                    "adjusted_close",
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

        cached_at = self.cache_times.get(key)
        if (
            key in self.cache
            and cached_at is not None
            and time.time() - cached_at <= self.daily_cache_ttl_seconds
        ):
            return self.cache[key].copy()

        path = self._cache_path(*key)

        if path.exists():

            try:
                cached = pd.read_csv(
                    path,
                    parse_dates=["datetime"],
                )

                if (
                    self._age_seconds(path) <= self.daily_cache_ttl_seconds
                    and len(cached) >= min(
                    65,
                    int(outputsize),
                    )
                ):
                    self.cache[key] = cached
                    self.cache_times[key] = time.time()
                    return cached.copy()

            except Exception:
                pass

        api_symbol = self._api_symbol(symbol)

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
        self.cache_times[key] = time.time()

        return df.copy()

    def _quote_from_payload(self, data: dict, fallback_symbol: str) -> dict:
        api_symbol = self._api_symbol(data.get("code") or fallback_symbol)
        price = self._number(
            data.get("close")
            or data.get("price")
            or data.get("last")
        )

        if price is None or price <= 0:
            raise RuntimeError(f"Prezzo EODHD non disponibile per {api_symbol}.")

        timestamp_value = data.get("timestamp")
        observed_at = None
        try:
            observed_at = datetime.fromtimestamp(
                int(timestamp_value),
                tz=timezone.utc,
            ).isoformat()
        except (TypeError, ValueError, OSError):
            # Non inventiamo l'orario corrente quando il provider non invia
            # un timestamp: l'app mostrerà esplicitamente che non è verificato.
            observed_at = None

        quote = {
            "price": price,
            "previous_close": self._number(
                data.get("previousClose") or data.get("previous_close")
            ),
            "change_pct": self._number(
                data.get("change_p") or data.get("changePercent")
            ),
            "observed_at": observed_at,
            "source": "eodhd_live_or_delayed",
            "provider_ticker": api_symbol,
            "is_delayed": True,
        }
        return quote

    def realtime_trade_quote(self, symbol: str, timeout: float = 2.5) -> dict:
        """Return a true real-time US trade from the EODHD WebSocket feed.

        This method is intentionally on-demand (single detail/chart requests),
        never used by the global scanner. The user's All-In-One plan supports
        up to 50 concurrent US symbols and the WebSocket feed does not consume
        REST API calls. Non-US listings continue to use the documented delayed
        REST quote feed.
        """
        api_symbol = self._api_symbol(symbol)
        if not api_symbol.endswith(".US"):
            raise RuntimeError("Tempo reale WebSocket disponibile solo per azioni USA.")

        try:
            from websocket import create_connection
        except Exception as error:
            raise RuntimeError("Client WebSocket non installato sul server.") from error

        plain_symbol = api_symbol.split(".", 1)[0]
        ws = None
        deadline = time.time() + max(0.8, float(timeout))
        try:
            ws = create_connection(
                f"wss://ws.eodhistoricaldata.com/ws/us?api_token={self.api_key}",
                timeout=max(0.8, float(timeout)),
            )
            # EODHD sends an authorization status first. Read and validate it
            # before subscribing so intermittent socket timing cannot turn the
            # first status frame into an apparent quote timeout.
            auth_raw = ws.recv()
            try:
                auth_payload = json.loads(auth_raw)
            except (TypeError, ValueError):
                auth_payload = {}
            if (
                isinstance(auth_payload, dict)
                and auth_payload.get("status_code")
                and int(auth_payload.get("status_code")) != 200
            ):
                raise RuntimeError(
                    str(auth_payload.get("message") or "WebSocket EODHD non autorizzato.")
                )

            ws.send(json.dumps({"action": "subscribe", "symbols": plain_symbol}))
            while time.time() < deadline:
                raw = ws.recv()
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("status_code") and int(payload.get("status_code")) != 200:
                    raise RuntimeError(str(payload.get("message") or "WebSocket EODHD non autorizzato."))
                if str(payload.get("s") or "").upper() != plain_symbol:
                    continue
                price = self._number(payload.get("p"))
                if price is None or price <= 0:
                    continue
                observed_at = None
                try:
                    observed_at = datetime.fromtimestamp(
                        int(payload.get("t")) / 1000.0, tz=timezone.utc
                    ).isoformat()
                except (TypeError, ValueError, OSError):
                    pass
                return {
                    "price": price,
                    "previous_close": None,
                    "change_pct": None,
                    "observed_at": observed_at,
                    "source": "eodhd_websocket_realtime",
                    "provider_ticker": api_symbol,
                    "is_delayed": False,
                    "market_status": str(payload.get("ms") or "").lower(),
                }
            raise RuntimeError("Nessun trade realtime ricevuto entro il timeout.")
        except RuntimeError:
            raise
        except Exception as error:
            # Never include the connection URL in a client-facing error: it
            # contains the private API token in the query string.
            raise RuntimeError("WebSocket EODHD realtime temporaneamente non disponibile.") from error
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    def latest_quote(self, symbol: str) -> dict:
        """Prezzo più recente disponibile, mai ricavato da una cache eterna."""
        if self._capability_blocked(self._quotes_denied_at):
            raise RuntimeError(
                "EODHD: quote live/ritardate non incluse nel piano (HTTP 403)."
            )
        api_symbol = self._api_symbol(symbol)
        cached = self.quote_cache.get(api_symbol)

        if cached and time.time() - cached[0] <= self.live_quote_ttl_seconds:
            return dict(cached[1])

        try:
            data = self._request(f"real-time/{api_symbol}")
        except RuntimeError as error:
            if "HTTP 403" in str(error):
                self._quotes_denied_at = time.time()
            raise

        if not isinstance(data, dict):
            raise RuntimeError(f"Quota EODHD non valida per {api_symbol}.")

        quote = self._quote_from_payload(data, api_symbol)
        self.quote_cache[api_symbol] = (time.time(), quote)
        return dict(quote)

    def batch_latest_quotes(
        self,
        symbols: Iterable[str],
        batch_size: int = 20,
    ) -> dict[str, dict]:
        """Recupera quote in gruppi, mantenendo una chiave per ticker provider."""
        normalized = list(dict.fromkeys(self._api_symbol(value) for value in symbols))
        output: dict[str, dict] = {}
        pending = []
        now = time.time()

        for symbol in normalized:
            cached = self.quote_cache.get(symbol)
            if cached and now - cached[0] <= self.live_quote_ttl_seconds:
                output[symbol] = dict(cached[1])
            else:
                pending.append(symbol)

        if self._capability_blocked(self._quotes_denied_at):
            return output

        size = max(1, min(int(batch_size), 20))
        for start in range(0, len(pending), size):
            chunk = pending[start:start + size]
            if not chunk:
                continue
            params = {"s": ",".join(chunk[1:])} if len(chunk) > 1 else None
            try:
                data = self._request(f"real-time/{chunk[0]}", params=params)
            except RuntimeError as error:
                if "HTTP 403" in str(error):
                    self._quotes_denied_at = time.time()
                    break
                continue

            payloads = data if isinstance(data, list) else [data]
            for payload in payloads:
                if not isinstance(payload, dict):
                    continue
                try:
                    fallback = str(payload.get("code") or chunk[0])
                    quote = self._quote_from_payload(payload, fallback)
                    key = str(quote["provider_ticker"]).upper()
                    self.quote_cache[key] = (time.time(), quote)
                    output[key] = dict(quote)
                except Exception:
                    continue

        return output

    def currency_to_usd_rate(self, currency: str) -> float | None:
        """Cambio EOD più recente per rendere confrontabili le capitalizzazioni.

        Il valore originale nella valuta di quotazione non viene mai
        sovrascritto: questo tasso alimenta soltanto i campi espliciti *_usd.
        """
        code = str(currency or "").strip().upper()
        if code == "USD":
            return 1.0

        direct = {"EUR", "GBP", "AUD", "NZD"}
        inverse = {
            "JPY", "CAD", "CHF", "HKD", "SEK", "NOK", "DKK", "ZAR",
            "PLN", "CNY", "SGD", "INR", "MXN", "BRL", "KRW", "TWD",
        }
        if code in direct:
            symbol = f"{code}USD.FOREX"
            invert = False
        elif code in inverse:
            symbol = f"USD{code}.FOREX"
            invert = True
        else:
            return None

        try:
            history = self.daily_history(symbol, outputsize=5)
            value = float(history["close"].dropna().iloc[-1])
            if value <= 0:
                return None
            return (1.0 / value) if invert else value
        except Exception:
            return None

    @staticmethod
    def _bulk_exchange_code(exchange: str) -> str:
        code = str(exchange or "US").strip().upper()
        if code in {"NASDAQ", "NYSE", "AMEX", "BATS", "NYSE MKT", "NYSE ARCA"}:
            return "US"
        if code == "AX":
            return "AU"
        return code

    @classmethod
    def _bulk_suffix(cls, exchange: str) -> str:
        return cls._bulk_exchange_code(exchange)

    @staticmethod
    def _bulk_currency(exchange: str) -> str:
        code = str(exchange or "").strip().upper()
        return {
            "NASDAQ": "USD", "NYSE": "USD", "AMEX": "USD", "BATS": "USD", "US": "USD",
            "LSE": "GBP", "TO": "CAD", "V": "CAD", "PA": "EUR", "XETRA": "EUR",
            "F": "EUR", "MI": "EUR", "AS": "EUR", "BR": "EUR", "MC": "EUR",
            "LS": "EUR", "SW": "CHF", "ST": "SEK", "CO": "DKK", "HE": "EUR",
            "OL": "NOK", "TSE": "JPY", "HK": "HKD", "AU": "AUD", "JSE": "ZAR",
            "WAR": "PLN",
        }.get(code, "")

    @staticmethod
    def _light_score(*, drawdown_pct, change_pct, volume_ratio, ema50_pct, ema200_pct) -> float:
        def finite(value):
            try:
                value = float(value)
                return value if pd.notna(value) else None
            except (TypeError, ValueError):
                return None

        dd = finite(drawdown_pct)
        ch = finite(change_pct)
        vr = finite(volume_ratio)
        e50 = finite(ema50_pct)
        e200 = finite(ema200_pct)

        draw_score = 0.0 if dd is None else max(0.0, min(100.0, ((-dd) - 8.0) / 52.0 * 100.0))
        shock_score = 0.0 if ch is None else max(0.0, min(100.0, ((-ch) - 1.0) / 14.0 * 100.0))
        volume_score = 0.0 if vr is None else max(0.0, min(100.0, (vr - 1.0) * 55.0))
        trend_parts = []
        for distance in (e50, e200):
            if distance is not None:
                trend_parts.append(max(0.0, min(100.0, (-distance) / 35.0 * 100.0)))
        trend_score = sum(trend_parts) / len(trend_parts) if trend_parts else 0.0
        return round(
            max(0.0, min(100.0,
                draw_score * 0.55
                + shock_score * 0.25
                + volume_score * 0.12
                + trend_score * 0.08
            )),
            1,
        )

    def bulk_eod_extended(self, exchange: str) -> pd.DataFrame:
        """Whole-exchange EOD snapshot with technical fields.

        EODHD charges a flat bulk cost per exchange, so this is the preferred
        first-level scanner: tens of thousands of listings can be checked
        without one HTTP request per ticker.
        """
        code = self._bulk_exchange_code(exchange)
        if not code:
            return pd.DataFrame()
        cache_path = self._bulk_eod_cache_path(code)
        data = None
        if cache_path.exists() and self._age_seconds(cache_path) <= self.bulk_cache_ttl_seconds:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                data = None
        if not isinstance(data, list):
            data = self._request(
                f"eod-bulk-last-day/{code}",
                params={"filter": "extended"},
            )
            if not isinstance(data, list):
                raise RuntimeError(f"EODHD: bulk EOD non valido per {code}.")
            try:
                cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return pd.DataFrame(data)

    def bulk_market_universe(
        self,
        exchanges: Iterable[str],
        *,
        min_avg_volume: int = 50_000,
        min_price: float = 1.0,
        min_market_cap_usd: float = 500_000_000,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Scan a broad global common-stock universe with one bulk pull/exchange."""
        rows: list[dict] = []
        raw_rows = 0
        scanned_exchanges: list[str] = []
        errors: dict[str, str] = {}
        rates: dict[str, float | None] = {"USD": 1.0}
        symbol_metadata: dict[str, dict[str, dict]] = {}

        def metadata_for(exchange_code: str) -> dict[str, dict]:
            # Venue-specific US bulk endpoints (NASDAQ/NYSE/AMEX/BATS) still
            # use the composite US exchange-symbol catalogue for currency and
            # venue metadata. Other exchanges use their own catalogue.
            key = "US" if exchange_code in {"NASDAQ", "NYSE", "AMEX", "BATS", "NYSE MKT"} else exchange_code
            if key not in symbol_metadata:
                try:
                    symbols = self.exchange_symbols(
                        key, include_delisted=False, common_stocks_only=False
                    )
                    symbol_metadata[key] = {
                        str(item.get("ticker") or "").upper(): item
                        for item in symbols
                        if item.get("ticker")
                    }
                except Exception:
                    symbol_metadata[key] = {}
            return symbol_metadata[key]

        for requested in tuple(exchanges):
            exchange = self._bulk_exchange_code(requested)
            if not exchange or exchange in scanned_exchanges:
                continue
            try:
                frame = self.bulk_eod_extended(exchange)
            except Exception as error:
                errors[exchange] = str(error)
                continue
            if frame is None or frame.empty:
                continue
            scanned_exchanges.append(exchange)
            raw_rows += len(frame)
            meta_map = metadata_for(exchange)
            default_currency = self._bulk_currency(exchange)
            suffix = self._bulk_suffix(exchange)

            for _, item in frame.iterrows():
                asset_type = str(item.get("type") or "").strip()
                if asset_type.lower().replace("_", " ") not in {"common stock", "commonstock", "stock"}:
                    continue
                code = str(item.get("code") or "").strip().upper()
                if not code:
                    continue
                meta = meta_map.get(code, {})
                venue = str(
                    meta.get("venue")
                    or item.get("exchange_short_name")
                    or item.get("exchange")
                    or exchange
                ).strip().upper()
                if exchange == "US" and venue not in {
                    "NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS", "US"
                }:
                    # Keep OTC/PINK and other non-core US venues out of the
                    # automatic scanner. They remain available to manual search.
                    continue
                currency = self._normalize_currency(
                    meta.get("currency") or item.get("currency") or default_currency
                )
                if currency and currency not in rates:
                    fx_currency = "GBP" if currency == "GBX" else currency
                    rates[currency] = self.currency_to_usd_rate(fx_currency)
                fx_rate = rates.get(currency)
                close = self._number(item.get("adjusted_close") or item.get("close"))
                market_cap = self._number(item.get("MarketCapitalization") or item.get("market_capitalization"))
                avgvol = self._number(item.get("avgvol_50d") or item.get("avgvol_200d") or item.get("avgvol_14d"))
                if close is None or close < float(min_price):
                    continue
                if avgvol is None or avgvol < int(min_avg_volume):
                    continue
                market_cap_usd = (
                    market_cap * self._major_currency_factor(currency) * fx_rate
                    if market_cap is not None and fx_rate else None
                )
                if market_cap_usd is None or market_cap_usd < float(min_market_cap_usd):
                    continue

                high250 = self._number(item.get("hi_250d"))
                ema50 = self._number(item.get("ema_50d"))
                ema200 = self._number(item.get("ema_200d"))
                volume = self._number(item.get("volume"))
                avg14 = self._number(item.get("avgvol_14d"))
                change_p = self._number(item.get("change_p"))
                drawdown = ((close / high250) - 1.0) * 100.0 if high250 and high250 > 0 else None
                volume_ratio = (volume / avg14) if volume is not None and avg14 and avg14 > 0 else None
                ema50_pct = ((close / ema50) - 1.0) * 100.0 if ema50 and ema50 > 0 else None
                ema200_pct = ((close / ema200) - 1.0) * 100.0 if ema200 and ema200 > 0 else None
                light_score = self._light_score(
                    drawdown_pct=drawdown,
                    change_pct=change_p,
                    volume_ratio=volume_ratio,
                    ema50_pct=ema50_pct,
                    ema200_pct=ema200_pct,
                )
                major_factor = self._major_currency_factor(currency)
                liquidity_usd = (
                    close * avgvol * major_factor * fx_rate
                    if close is not None and avgvol is not None and fx_rate
                    else None
                )
                rows.append({
                    "ticker": f"{code}.{suffix}",
                    "display_ticker": code,
                    "company": str(item.get("name") or code),
                    "asset_type": asset_type or "Common Stock",
                    "light_exchange": suffix,
                    "source_exchange": exchange,
                    "currency": currency,
                    "country": str(meta.get("country") or ""),
                    "venue": venue,
                    "isin": str(meta.get("isin") or "").strip().upper(),
                    "light_liquidity_usd": liquidity_usd,
                    "light_data_date": item.get("date"),
                    "light_last_price": close,
                    "light_return_1d_pct": change_p,
                    "light_market_cap": market_cap,
                    "light_market_cap_usd": market_cap_usd,
                    "market_cap_fx_to_usd": fx_rate,
                    "light_volume": avgvol,
                    "light_session_volume": volume,
                    "light_volume_ratio": volume_ratio,
                    "light_drawdown_250d_pct": drawdown,
                    "light_hi_250d": high250,
                    "light_lo_250d": self._number(item.get("lo_250d")),
                    "light_ema50_pct": ema50_pct,
                    "light_ema200_pct": ema200_pct,
                    "light_anomaly_score": light_score,
                })

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.drop_duplicates(subset=["ticker"], keep="first")

            # Cross-listed companies can appear on several exchanges. When an
            # ISIN is available, keep the most liquid listing as the Light
            # representative so scarce Deep slots are not wasted on duplicates.
            # Rows without an ISIN remain distinct.
            result["_entity_key"] = result.apply(
                lambda row: (
                    f"ISIN:{str(row.get('isin') or '').strip().upper()}"
                    if str(row.get("isin") or "").strip()
                    else f"TICKER:{str(row.get('ticker') or '').strip().upper()}"
                ),
                axis=1,
            )
            result = result.sort_values(
                ["light_liquidity_usd", "light_market_cap_usd", "light_anomaly_score"],
                ascending=[False, False, False],
                na_position="last",
            ).drop_duplicates(subset=["_entity_key"], keep="first")
            result = result.drop(columns=["_entity_key"], errors="ignore")
            result = result.sort_values(
                ["light_anomaly_score", "light_market_cap_usd"],
                ascending=[False, False],
                na_position="last",
            )
            if max_rows is not None and int(max_rows) > 0:
                result = result.head(int(max_rows))
            result = result.reset_index(drop=True)
        self.last_bulk_stats = {
            "exchanges_requested": len(tuple(exchanges)),
            "exchanges_scanned": scanned_exchanges,
            "raw_rows": int(raw_rows),
            "eligible_rows": int(len(result)),
            "estimated_api_calls": int(len(scanned_exchanges) * 100),
            "errors": errors,
        }
        return result

    def press_releases(self, symbol: str, limit: int = 12, days: int = 45) -> list[dict]:
        """EODHD financial news normalized for the catalyst classifier."""
        api_symbol = self._api_symbol(symbol)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=max(1, int(days)))
        data = self._request(
            "news",
            params={
                "s": api_symbol,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "limit": max(1, min(int(limit), 100)),
                "offset": 0,
            },
        )
        if not isinstance(data, list):
            return []
        output = []
        for item in data:
            if not isinstance(item, dict):
                continue
            output.append({
                "title": item.get("title") or "",
                "text": item.get("content") or item.get("description") or "",
                "summary": item.get("content") or item.get("description") or "",
                "date": item.get("date"),
                "link": item.get("link"),
                "source": item.get("source") or "EODHD news",
                "sentiment": item.get("sentiment"),
            })
        return output

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

        exchanges = tuple(exchanges)
        rows: list[dict] = []
        requests_used = 0
        exchanges_scanned = []

        exchange_currencies = {
            "us": "USD", "nyse": "USD", "nasdaq": "USD", "amex": "USD",
            "lse": "GBP", "to": "CAD", "v": "CAD", "pa": "EUR",
            "xetra": "EUR", "mi": "EUR", "as": "EUR", "br": "EUR",
            "mc": "EUR", "ls": "EUR", "sw": "CHF", "st": "SEK",
            "co": "DKK", "he": "EUR", "ol": "NOK", "tse": "JPY",
            "hk": "HKD", "au": "AUD", "ax": "AUD", "jse": "ZAR",
        }

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
                    "avgvol_200d",
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
                currency = exchange_currencies.get(str(exchange).lower())
                rate = self.currency_to_usd_rate(currency) if currency else None
                local_floor = (
                    float(min_market_cap) / rate
                    if rate is not None and rate > 0
                    else float(min_market_cap)
                )
                filters.append(
                    [
                        "market_capitalization",
                        ">=",
                        local_floor,
                    ]
                )

            desired = max(1, min(int(limit_per_exchange), 1000))
            offset = 0
            exchange_rows = []

            while len(exchange_rows) < desired:
                if requests_used >= self.screener_max_requests:
                    break
                page_size = min(500, desired - len(exchange_rows))
                data = self._request(
                    "screener",
                    params={
                        "filters": json.dumps(filters, separators=(",", ":")),
                        "sort": "refund_1d_p.asc",
                        "limit": page_size,
                        "offset": offset,
                    },
                )
                requests_used += 1

                if isinstance(data, dict):
                    data = data.get("data", [])
                if not isinstance(data, list) or not data:
                    break

                exchange_rows.extend(data)
                offset += len(data)
                if len(data) < page_size:
                    break

            if exchange_rows:
                exchanges_scanned.append(str(exchange).lower())
                rows.extend(exchange_rows)
            if requests_used >= self.screener_max_requests:
                break

        self.last_screener_stats = {
            "requests_used": requests_used,
            "exchanges_requested": len(exchanges),
            "exchanges_scanned": exchanges_scanned,
            "candidates_returned": len(rows),
        }

        return pd.DataFrame(rows)

    def screen_market_sample(
        self,
        exchanges: Iterable[str] = ("us",),
        min_avg_volume: int = 200_000,
        min_price: float = 2.0,
        limit_per_exchange: int = 2,
    ) -> pd.DataFrame:
        """Neutral multi-market sample ordered by market capitalization.

        Unlike screen_candidates(), this method does NOT filter on daily returns.
        It is used only by the Global Market Tension Engine to avoid selecting
        securities because they already moved sharply.
        """
        rows: list[dict] = []
        requests_used = 0
        for exchange in tuple(exchanges):
            if requests_used >= self.screener_max_requests:
                break
            filters = [
                ["exchange", "=", str(exchange).lower()],
                ["avgvol_200d", ">=", int(min_avg_volume)],
                ["adjusted_close", ">=", float(min_price)],
            ]
            desired = max(1, min(int(limit_per_exchange), 25))
            data = self._request(
                "screener",
                params={
                    "filters": json.dumps(filters, separators=(",", ":")),
                    "sort": "market_capitalization.desc",
                    "limit": desired,
                    "offset": 0,
                },
            )
            requests_used += 1
            if isinstance(data, dict):
                data = data.get("data", [])
            if isinstance(data, list):
                rows.extend(data[:desired])
        return pd.DataFrame(rows)

    def exchange_symbols(
        self,
        exchange: str,
        *,
        include_delisted: bool = False,
        common_stocks_only: bool = True,
        max_age_hours: int = 24,
    ) -> list[dict]:
        """Restituisce un universo neutrale di titoli dell'exchange.

        Questo endpoint non filtra in base al movimento del giorno: viene usato
        dal backfill storico per evitare di imparare solamente dai ribassi che
        lo Screener live ha gia selezionato. La lista viene memorizzata per 24
        ore per non consumare richieste inutili.
        """
        code = str(exchange or "").strip().upper()
        if not code:
            return []

        cache_path = self._exchange_symbols_cache_path(code)
        data = None
        if (
            cache_path.exists()
            and self._age_seconds(cache_path) <= max(1, int(max_age_hours)) * 3600
        ):
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                data = None

        if not isinstance(data, list):
            data = self._request(
                f"exchange-symbol-list/{code}",
                params={"delisted": 1 if include_delisted else 0},
            )
            if not isinstance(data, list):
                return []
            try:
                cache_path.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass

        allowed_us_venues = {
            "NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS",
        }
        output = []
        seen = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("Code") or item.get("code") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            asset_type = str(item.get("Type") or item.get("type") or "").strip()
            normalized_type = asset_type.lower().replace("_", " ")
            if common_stocks_only and normalized_type not in {
                "common stock", "commonstock", "stock",
            }:
                continue
            venue = str(item.get("Exchange") or item.get("exchange") or code).strip().upper()
            if code == "US" and venue not in allowed_us_venues:
                continue
            seen.add(ticker)
            output.append({
                "ticker": ticker,
                "provider_ticker": f"{ticker}.{code}",
                "company": str(item.get("Name") or item.get("name") or ticker),
                "exchange": code,
                "venue": venue,
                "currency": self._normalize_currency(
                    item.get("Currency") or item.get("currency") or ""
                ),
                "country": str(item.get("Country") or item.get("country") or ""),
                "isin": str(
                    item.get("ISIN") or item.get("Isin") or item.get("isin") or ""
                ).strip().upper(),
                "type": asset_type or "Common Stock",
            })
        return output

    def search_symbols(
        self,
        query: str,
        limit: int = 12,
    ) -> list[dict]:
        text = str(query).strip()

        if not text:
            return []

        data = self._request(
            f"search/{text}",
            params={
                "limit": max(1, min(int(limit), 25)),
            },
        )

        if not isinstance(data, list):
            return []

        results = []

        for item in data:
            if not isinstance(item, dict):
                continue

            code = str(item.get("Code") or item.get("code") or "").upper()
            exchange = str(
                item.get("Exchange")
                or item.get("exchange")
                or "US"
            ).upper()
            name = str(item.get("Name") or item.get("name") or code)
            item_type = str(item.get("Type") or item.get("type") or "")

            if not code:
                continue

            provider_ticker = code if "." in code else f"{code}.{exchange}"

            results.append({
                "ticker": code.split(".")[0],
                "provider_ticker": provider_ticker,
                "company": name,
                "exchange": exchange,
                "venue": str(item.get("Exchange") or item.get("exchange") or exchange),
                "currency": self._normalize_currency(
                    item.get("Currency") or item.get("currency") or item.get("CurrencyCode")
                ),
                "country": str(item.get("Country") or item.get("country") or ""),
                "isin": str(item.get("ISIN") or item.get("isin") or ""),
                "is_primary": bool(item.get("isPrimary") or item.get("IsPrimary")),
                "previous_close": self._number(item.get("previousClose") or item.get("previous_close")),
                "previous_close_date": item.get("previousCloseDate") or item.get("previous_close_date"),
                "type": item_type,
            })

        return results[: int(limit)]

    def intraday_history(
        self,
        symbol: str,
        days: int = 1,
        interval: str = "5m",
    ) -> pd.DataFrame:
        if self._capability_blocked(self._intraday_denied_at):
            raise RuntimeError(
                "EODHD: storico intraday non incluso nel piano (HTTP 403)."
            )
        api_symbol = self._api_symbol(symbol)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, int(days)))

        try:
            data = self._request(
                f"intraday/{api_symbol}",
                params={
                    "interval": interval,
                    "from": int(start.timestamp()),
                    "to": int(end.timestamp()),
                },
            )
        except RuntimeError as error:
            if "HTTP 403" in str(error):
                self._intraday_denied_at = time.time()
            raise

        if not isinstance(data, list) or not data:
            raise RuntimeError(
                f"Storico intraday non disponibile per {api_symbol}."
            )

        frame = pd.DataFrame(data)

        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(
                frame["datetime"],
                errors="coerce",
                utc=True,
            )
        elif "timestamp" in frame.columns:
            frame["datetime"] = pd.to_datetime(
                frame["timestamp"],
                unit="s",
                errors="coerce",
                utc=True,
            )
        else:
            raise RuntimeError(
                f"Risposta intraday non valida per {api_symbol}."
            )

        for column in ["open", "high", "low", "close", "volume"]:
            if column not in frame.columns:
                frame[column] = 0 if column == "volume" else None

            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame["volume"] = frame["volume"].fillna(0.0)
        for column in ["open", "high", "low"]:
            frame[column] = frame[column].fillna(frame["close"])

        frame = (
            frame.dropna(subset=["datetime", "close"])
            .sort_values("datetime")
            [["datetime", "open", "high", "low", "close", "volume"]]
            .reset_index(drop=True)
        )

        if frame.empty:
            raise RuntimeError(
                f"Storico intraday vuoto per {api_symbol}."
            )

        return frame
