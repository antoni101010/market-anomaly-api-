import requests
import os
import time
from config import CONFIG

class SecEdgarProvider:
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, user_agent=None, timeout=20, min_interval=0.12):
        self.user_agent = user_agent or CONFIG.sec_user_agent
        self.timeout = timeout
        self.min_interval = float(min_interval)
        self._last_request = 0.0
        self._ticker_map = None
        self._facts_cache = {}

    def _headers(self):
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def _get(self, url):
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0: time.sleep(wait)
        r = requests.get(url, headers=self._headers(), timeout=self.timeout)
        self._last_request = time.monotonic()
        r.raise_for_status()
        return r

    def ticker_to_cik(self):
        if self._ticker_map is None:
            r = self._get(self.TICKERS_URL)
            raw = r.json()
            self._ticker_map = {
                row["ticker"].upper(): str(row["cik_str"]).zfill(10)
                for row in raw.values()
            }
        return self._ticker_map

    def companyfacts(self, ticker):
        ticker = ticker.upper()
        if ticker in self._facts_cache:
            return self._facts_cache[ticker]
        cik = self.ticker_to_cik().get(ticker)
        if not cik:
            return None
        r = self._get(self.FACTS_URL.format(cik=cik))
        facts = r.json()
        self._facts_cache[ticker] = facts
        return facts

    @staticmethod
    def _annual_values(facts, tags, unit="USD", n=3):
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            node = usgaap.get(tag)
            if not node:
                continue
            rows = node.get("units", {}).get(unit, [])
            rows = [x for x in rows if x.get("form") in {"10-K","10-K/A"} and x.get("fy") and x.get("val") is not None]
            by_fy = {}
            for x in rows:
                fy = int(x["fy"])
                if fy not in by_fy or str(x.get("filed","")) > str(by_fy[fy].get("filed","")):
                    by_fy[fy] = x
            ordered = [by_fy[k] for k in sorted(by_fy)]
            vals = [float(x["val"]) for x in ordered[-n:]]
            if vals:
                return vals
        return []

    @staticmethod
    def _latest_value(facts, tags, units=("shares","USD")):
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            node = usgaap.get(tag)
            if not node:
                continue
            for unit in units:
                rows = node.get("units", {}).get(unit, [])
                rows = [x for x in rows if x.get("val") is not None and x.get("filed")]
                if rows:
                    rows = sorted(rows, key=lambda x: (str(x.get("filed","")), str(x.get("end",""))))
                    return float(rows[-1]["val"])
        return None

    def metrics(self, ticker, last_price=None):
        facts = self.companyfacts(ticker)
        if not facts:
            return {}

        revenues = self._annual_values(
            facts,
            ["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","SalesRevenueNet"],
            "USD", 3
        )
        net_income = self._annual_values(facts, ["NetIncomeLoss"], "USD", 3)
        assets = self._annual_values(facts, ["Assets"], "USD", 2)
        liabilities = self._annual_values(facts, ["Liabilities"], "USD", 2)
        cfo = self._annual_values(
            facts,
            ["NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
            "USD", 2
        )
        capex = self._annual_values(
            facts,
            ["PaymentsToAcquirePropertyPlantAndEquipment","PaymentsForAdditionsToPropertyPlantAndEquipment"],
            "USD", 2
        )

        revenue_growth = None
        if len(revenues) >= 2 and revenues[-2] != 0:
            revenue_growth = (revenues[-1]/revenues[-2]-1)*100

        net_margin = None
        if revenues and net_income and revenues[-1] != 0:
            net_margin = net_income[-1]/revenues[-1]*100

        liabilities_to_assets = None
        if assets and liabilities and assets[-1] != 0:
            liabilities_to_assets = liabilities[-1]/assets[-1]

        fcf_margin = None
        if revenues and cfo:
            cap = capex[-1] if capex else 0
            fcf = cfo[-1] - cap
            if revenues[-1]:
                fcf_margin = fcf/revenues[-1]*100

        shares = self._latest_value(
            facts,
            ["CommonStocksIncludingAdditionalPaidInCapitalMember",
             "EntityCommonStockSharesOutstanding",
             "CommonStockSharesOutstanding"],
            units=("shares",)
        )

        approx_pe = None
        approx_ps = None
        if last_price and shares and shares > 0:
            market_cap = last_price * shares
            if net_income and net_income[-1] > 0:
                approx_pe = market_cap/net_income[-1]
            if revenues and revenues[-1] > 0:
                approx_ps = market_cap/revenues[-1]

        return {
            "revenue_growth_pct": revenue_growth,
            "net_margin_pct": net_margin,
            "liabilities_to_assets": liabilities_to_assets,
            "fcf_margin_pct": fcf_margin,
            "approx_pe": approx_pe,
            "approx_ps": approx_ps,
        }


    def recent_filings(self, ticker, forms=("8-K","10-Q","10-K"), limit=8):
        cik = self.ticker_to_cik().get(ticker.upper())
        if not cik:
            return []
        r = self._get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms_col = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession = recent.get("accessionNumber", [])
        primary = recent.get("primaryDocument", [])

        out = []
        for i, form in enumerate(forms_col):
            if form not in forms:
                continue
            acc = accession[i] if i < len(accession) else None
            doc = primary[i] if i < len(primary) else None
            url = None
            if acc and doc:
                acc_clean = acc.replace("-", "")
                cik_int = str(int(cik))
                url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc}"
            out.append({
                "form": form,
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                "accession": acc,
                "url": url,
            })
            if len(out) >= int(limit):
                break
        return out


    @staticmethod
    def _annual_values_as_of(facts, tags, as_of, unit="USD", n=3):
        import pandas as pd
        as_of = pd.Timestamp(as_of).normalize()
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        if isinstance(tags, str): tags=[tags]
        for tag in tags:
            node=usgaap.get(tag)
            if not node: continue
            rows=node.get("units",{}).get(unit,[])
            usable=[]
            for x in rows:
                if x.get("form") not in {"10-K","10-K/A"}: continue
                if x.get("fy") is None or x.get("val") is None or not x.get("filed"): continue
                try: filed=pd.Timestamp(x["filed"]).normalize()
                except Exception: continue
                if filed<=as_of: usable.append(x)
            by_fy={}
            for x in usable:
                fy=int(x[
