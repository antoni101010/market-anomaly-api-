import time

import pandas as pd
import requests

from config import CONFIG


class SecEdgarProvider:
    TICKERS_URL = (
        "https://www.sec.gov/files/"
        "company_tickers.json"
    )

    FACTS_URL = (
        "https://data.sec.gov/api/xbrl/"
        "companyfacts/CIK{cik}.json"
    )

    def __init__(
        self,
        user_agent=None,
        timeout=20,
        min_interval=0.12,
    ):
        self.user_agent = (
            user_agent
            or CONFIG.sec_user_agent
        )
        self.timeout = timeout
        self.min_interval = float(
            min_interval
        )
        self._last_request = 0.0
        self._ticker_map = None
        self._facts_cache = {}

    def _headers(self):
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": (
                "gzip, deflate"
            ),
        }

    def _get(self, url):
        wait = (
            self.min_interval
            - (
                time.monotonic()
                - self._last_request
            )
        )

        if wait > 0:
            time.sleep(wait)

        response = requests.get(
            url,
            headers=self._headers(),
            timeout=self.timeout,
        )

        self._last_request = (
            time.monotonic()
        )

        response.raise_for_status()

        return response

    def ticker_to_cik(self):
        if self._ticker_map is None:
            response = self._get(
                self.TICKERS_URL
            )

            raw = response.json()

            self._ticker_map = {
                row["ticker"].upper(): str(
                    row["cik_str"]
                ).zfill(10)
                for row in raw.values()
            }

        return self._ticker_map

    def companyfacts(self, ticker):
        ticker = ticker.upper()

        if ticker in self._facts_cache:
            return self._facts_cache[
                ticker
            ]

        cik = self.ticker_to_cik().get(
            ticker
        )

        if not cik:
            return None

        response = self._get(
            self.FACTS_URL.format(
                cik=cik
            )
        )

        facts = response.json()

        self._facts_cache[
            ticker
        ] = facts

        return facts

    @staticmethod
    def _annual_values(
        facts,
        tags,
        unit="USD",
        number=3,
    ):
        usgaap = (
            facts
            .get("facts", {})
            .get("us-gaap", {})
        )

        if isinstance(tags, str):
            tags = [tags]

        for tag in tags:
            node = usgaap.get(tag)

            if not node:
                continue

            rows = (
                node
                .get("units", {})
                .get(unit, [])
            )

            rows = [
                row
                for row in rows
                if (
                    row.get("form")
                    in {"10-K", "10-K/A"}
                    and row.get("fy")
                    and row.get("val")
                    is not None
                )
            ]

            by_year = {}

            for row in rows:
                year = int(
                    row["fy"]
                )

                if (
                    year not in by_year
                    or str(
                        row.get(
                            "filed",
                            "",
                        )
                    )
                    > str(
                        by_year[year].get(
                            "filed",
                            "",
                        )
                    )
                ):
                    by_year[year] = row

            ordered = [
                by_year[key]
                for key in sorted(
                    by_year
                )
            ]

            values = [
                float(row["val"])
                for row
                in ordered[-number:]
            ]

            if values:
                return values

        return []

    @staticmethod
    def _latest_value(
        facts,
        tags,
        units=("shares", "USD"),
    ):
        """
        Cerca il dato più recente sia nel
        namespace SEC dei sia in us-gaap.

        EntityCommonStockSharesOutstanding
        viene normalmente pubblicato in dei.
        """

        namespaces = facts.get(
            "facts",
            {},
        )

        if isinstance(tags, str):
            tags = [tags]

        for namespace in (
            "dei",
            "us-gaap",
        ):
            nodes = namespaces.get(
                namespace,
                {},
            )

            for tag in tags:
                node = nodes.get(tag)

                if not node:
                    continue

                for unit in units:
                    rows = (
                        node
                        .get("units", {})
                        .get(unit, [])
                    )

                    rows = [
                        row
                        for row in rows
                        if (
                            row.get("val")
                            is not None
                            and row.get("filed")
                        )
                    ]

                    if rows:
                        rows = sorted(
                            rows,
                            key=lambda row: (
                                str(
                                    row.get(
                                        "filed",
                                        "",
                                    )
                                ),
                                str(
                                    row.get(
                                        "end",
                                        "",
                                    )
                                ),
                            ),
                        )

                        return float(
                            rows[-1]["val"]
                        )

        return None

    def metrics(
        self,
        ticker,
        last_price=None,
    ):
        facts = self.companyfacts(
            ticker
        )

        if not facts:
            return {}

        revenues = self._annual_values(
            facts,
            [
                (
                    "RevenueFromContractWith"
                    "CustomerExcludingAssessedTax"
                ),
                "Revenues",
                "SalesRevenueNet",
            ],
            "USD",
            3,
        )

        net_income = self._annual_values(
            facts,
            ["NetIncomeLoss"],
            "USD",
            3,
        )

        assets = self._annual_values(
            facts,
            ["Assets"],
            "USD",
            2,
        )

        liabilities = self._annual_values(
            facts,
            ["Liabilities"],
            "USD",
            2,
        )

        operating_cash = self._annual_values(
            facts,
            [
                (
                    "NetCashProvidedByUsedIn"
                    "OperatingActivities"
                ),
                (
                    "NetCashProvidedByUsedIn"
                    "OperatingActivities"
                    "ContinuingOperations"
                ),
            ],
            "USD",
            2,
        )

        capital_expenditure = (
            self._annual_values(
                facts,
                [
                    (
                        "PaymentsToAcquireProperty"
                        "PlantAndEquipment"
                    ),
                    (
                        "PaymentsForAdditionsTo"
                        "PropertyPlantAndEquipment"
                    ),
                ],
                "USD",
                2,
            )
        )

        revenue_growth = None

        if (
            len(revenues) >= 2
            and revenues[-2] != 0
        ):
            revenue_growth = (
                (
                    revenues[-1]
                    / revenues[-2]
                )
                - 1
            ) * 100

        net_margin = None

        if (
            revenues
            and net_income
            and revenues[-1] != 0
        ):
            net_margin = (
                net_income[-1]
                / revenues[-1]
                * 100
            )

        liabilities_to_assets = None

        if (
            assets
            and liabilities
            and assets[-1] != 0
        ):
            liabilities_to_assets = (
                liabilities[-1]
                / assets[-1]
            )

        free_cash_flow = None
        fcf_margin = None

        if revenues and operating_cash:
            capex = (
                capital_expenditure[-1]
                if capital_expenditure
                else 0
            )

            free_cash_flow = (
                operating_cash[-1]
                - capex
            )

            if revenues[-1]:
                fcf_margin = (
                    free_cash_flow
                    / revenues[-1]
                    * 100
                )

        shares = self._latest_value(
            facts,
            [
                (
                    "EntityCommonStock"
                    "SharesOutstanding"
                ),
                (
                    "CommonStockSharesOutstanding"
                ),
            ],
            units=("shares",),
        )

        approximate_pe = None
        approximate_ps = None
        market_cap = None
        fcf_yield_pct = None

        if (
            last_price is not None
            and shares is not None
            and shares > 0
        ):
            market_cap = (
                float(last_price)
                * shares
            )

            if (
                net_income
                and net_income[-1] > 0
            ):
                approximate_pe = (
                    market_cap
                    / net_income[-1]
                )

            if (
                revenues
                and revenues[-1] > 0
            ):
                approximate_ps = (
                    market_cap
                    / revenues[-1]
                )

            if (
                free_cash_flow is not None
                and market_cap > 0
            ):
                fcf_yield_pct = (
                    free_cash_flow
                    / market_cap
                    * 100
                )

        return {
            "revenue_growth_pct": (
                revenue_growth
            ),
            "net_margin_pct": (
                net_margin
            ),
            "liabilities_to_assets": (
                liabilities_to_assets
            ),
            "fcf_margin_pct": (
                fcf_margin
            ),
            "free_cash_flow_ttm": (
                free_cash_flow
            ),
            "shares_outstanding": (
                shares
            ),
            "approx_pe": (
                approximate_pe
            ),
            "approx_ps": (
                approximate_ps
            ),
            "pe_ratio": (
                approximate_pe
            ),
            "price_to_sales": (
                approximate_ps
            ),
            "market_cap": (
                market_cap
            ),
            "fcf_yield_pct": (
                fcf_yield_pct
            ),
            "fundamentals_source_sec": (
                True
            ),
        }

    def recent_filings(
        self,
        ticker,
        forms=(
            "8-K",
            "10-Q",
            "10-K",
        ),
        limit=8,
    ):
        cik = self.ticker_to_cik().get(
            ticker.upper()
        )

        if not cik:
            return []

        response = self._get(
            "https://data.sec.gov/"
            f"submissions/CIK{cik}.json"
        )

        data = response.json()

        recent = (
            data
            .get("filings", {})
            .get("recent", {})
        )

        if not recent:
            return []

        forms_column = recent.get(
            "form",
            [],
        )
        filing_dates = recent.get(
            "filingDate",
            [],
        )
        accessions = recent.get(
            "accessionNumber",
            [],
        )
        documents = recent.get(
            "primaryDocument",
            [],
        )

        output = []

        for index, form in enumerate(
            forms_column
        ):
            if form not in forms:
                continue

            accession = (
                accessions[index]
                if index < len(accessions)
                else None
            )

            document = (
                documents[index]
                if index < len(documents)
                else None
            )

            url = None

            if accession and document:
                clean_accession = (
                    accession.replace(
                        "-",
                        "",
                    )
                )

                cik_integer = str(
                    int(cik)
                )

                url = (
                    "https://www.sec.gov/"
                    "Archives/edgar/data/"
                    f"{cik_integer}/"
                    f"{clean_accession}/"
                    f"{document}"
                )

            output.append({
                "form": form,
                "filing_date": (
                    filing_dates[index]
                    if index
                    < len(filing_dates)
                    else None
                ),
                "accession": accession,
                "url": url,
            })

            if len(output) >= int(limit):
                break

        return output

    @staticmethod
    def _annual_values_as_of(
        facts,
        tags,
        as_of,
        unit="USD",
        number=3,
    ):
        as_of = pd.Timestamp(
            as_of
        ).normalize()

        usgaap = (
            facts
            .get("facts", {})
            .get("us-gaap", {})
        )

        if isinstance(tags, str):
            tags = [tags]

        for tag in tags:
            node = usgaap.get(tag)

            if not node:
                continue

            rows = (
                node
                .get("units", {})
                .get(unit, [])
            )

            usable = []

            for row in rows:
                if (
                    row.get("form")
                    not in {"10-K", "10-K/A"}
                ):
                    continue

                if (
                    row.get("fy") is None
                    or row.get("val") is None
                    or not row.get("filed")
                ):
                    continue

                try:
                    filed = pd.Timestamp(
                        row["filed"]
                    ).normalize()

                except Exception:
                    continue

                if filed <= as_of:
                    usable.append(row)

            by_year = {}

            for row in usable:
                year = int(
                    row["fy"]
                )

                if (
                    year not in by_year
                    or str(
                        row.get(
                            "filed",
                            "",
                        )
                    )
                    > str(
                        by_year[year].get(
                            "filed",
                            "",
                        )
                    )
                ):
                    by_year[year] = row

            ordered = [
                by_year[key]
                for key in sorted(
                    by_year
                )
            ]

            if ordered:
                return ordered[
                    -number:
                ]

        return []

    def metrics_as_of(
        self,
        ticker,
        as_of,
        last_price=None,
    ):
        """
        Fondamentali che utilizzano solamente
        dati pubblicati entro la data indicata.
        """

        facts = self.companyfacts(
            ticker
        )

        if not facts:
            return {}

        revenues = self._annual_values_as_of(
            facts,
            [
                (
                    "RevenueFromContractWith"
                    "CustomerExcludingAssessedTax"
                ),
                "Revenues",
                "SalesRevenueNet",
            ],
            as_of,
            "USD",
            3,
        )

        net_income = self._annual_values_as_of(
            facts,
            ["NetIncomeLoss"],
            as_of,
            "USD",
            3,
        )

        assets = self._annual_values_as_of(
            facts,
            ["Assets"],
            as_of,
            "USD",
            2,
        )

        liabilities = (
            self._annual_values_as_of(
                facts,
                ["Liabilities"],
                as_of,
                "USD",
                2,
            )
        )

        operating_cash = (
            self._annual_values_as_of(
                facts,
                [
                    (
                        "NetCashProvidedByUsedIn"
                        "OperatingActivities"
                    ),
                    (
                        "NetCashProvidedByUsedIn"
                        "OperatingActivities"
                        "ContinuingOperations"
                    ),
                ],
                as_of,
                "USD",
                2,
            )
        )

        capital_expenditure = (
            self._annual_values_as_of(
                facts,
                [
                    (
                        "PaymentsToAcquireProperty"
                        "PlantAndEquipment"
                    ),
                    (
                        "PaymentsForAdditionsTo"
                        "PropertyPlantAndEquipment"
                    ),
                ],
                as_of,
                "USD",
                2,
            )
        )

        def values(rows):
            return [
                float(row["val"])
                for row in rows
            ]

        revenue_values = values(
            revenues
        )
        income_values = values(
            net_income
        )
        asset_values = values(
            assets
        )
        liability_values = values(
            liabilities
        )
        cash_values = values(
            operating_cash
        )
        capex_values = values(
            capital_expenditure
        )

        revenue_growth = (
            (
                revenue_values[-1]
                / revenue_values[-2]
                - 1
            )
            * 100
            if (
                len(revenue_values) >= 2
                and revenue_values[-2]
            )
            else None
        )

        net_margin = (
            income_values[-1]
            / revenue_values[-1]
            * 100
            if (
                revenue_values
                and income_values
                and revenue_values[-1]
            )
            else None
        )

        liabilities_to_assets = (
            liability_values[-1]
            / asset_values[-1]
            if (
                asset_values
                and liability_values
                and asset_values[-1]
            )
            else None
        )

        fcf_margin = None

        if (
            revenue_values
            and cash_values
            and revenue_values[-1]
        ):
            fcf_margin = (
                cash_values[-1]
                - (
                    capex_values[-1]
                    if capex_values
                    else 0
                )
            ) / revenue_values[-1] * 100

        filed_dates = []

        for group in (
            revenues,
            net_income,
            assets,
            liabilities,
            operating_cash,
            capital_expenditure,
        ):
            filed_dates += [
                row.get("filed")
                for row in group
                if row.get("filed")
            ]

        return {
            "revenue_growth_pct": (
                revenue_growth
            ),
            "net_margin_pct": (
                net_margin
            ),
            "liabilities_to_assets": (
                liabilities_to_assets
            ),
            "fcf_margin_pct": (
                fcf_margin
            ),
            "pit_last_filed_date": (
                max(filed_dates)
                if filed_dates
                else None
            ),
            "pit_fact_count": sum(
                len(group)
                for group in (
                    revenues,
                    net_income,
                    assets,
                    liabilities,
                    operating_cash,
                    capital_expenditure,
                )
            ),
        }
