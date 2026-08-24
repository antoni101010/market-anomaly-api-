from __future__ import annotations

import json
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
