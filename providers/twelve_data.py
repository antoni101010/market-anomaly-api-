from __future__ import annotations
import hashlib
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd


class TwelveDataProvider:
    BASE = "https://api.twelvedata.com"

    def __init__(
        self,
        api_key,
        timeout=25,
        batch_size=8,
        cache_dir="data/price_cache",
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(
            1,
            min(int(batch_size), 8),
        )
        self.cache = {}
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        self.session.mount(
            "https://",
            HTTPAdapter(max_retries=retry),
        )

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else None
        )

        if self.cache_dir:
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _disk_key(
        self,
        symbol,
        outputsize,
        adjust,
    ):
        raw = (
            f"{symbol.upper()}|"
            f"{int(outputsize)}|"
            f"{adjust}"
        )

        return hashlib.sha1(
            raw.encode()
        ).hexdigest()[:20]

    def _disk_path(
        self,
        symbol,
        outputsize,
        adjust,
    ):
        if not self.cache_dir:
            return None

        key = self._disk_key(
            symbol,
            outputsize,
            adjust,
        )

        return self.cache_dir / (
            f"{symbol.upper()}_{key}.csv.gz"
        )

    def _load_disk(
        self,
        symbol,
        outputsize,
        adjust,
    ):
        path = self._disk_path(
            symbol,
            outputsize,
            adjust,
        )

        if not path or not path.exists():
            return None

        try:
            df = pd.read_csv(path)

            df["datetime"] = pd.to_datetime(
                df["datetime"],
                errors="coerce",
            )

            return (
                df.dropna(
                    subset=["datetime", "close"]
                )
                .sort_values("datetime")
                .reset_index(drop=True)
            )

        except Exception:
            return None

    def _save_disk(
        self,
        symbol,
        outputsize,
        adjust,
        df,
    ):
        path = self._disk_path(
            symbol,
            outputsize,
            adjust,
        )

        if path:
            try:
                df.to_csv(
                    path,
                    index=False,
                    compression="gzip",
                )
            except Exception:
                pass

    @staticmethod
    def _parse_item(item, symbol):
        if (
            isinstance(item, dict)
            and item.get("status") == "error"
        ):
            raise RuntimeError(
                item.get(
                    "message",
                    f"Errore Twelve Data per {symbol}",
                )
            )

        values = (
            item.get("values")
            if isinstance(item, dict)
            else None
        )

        if not values:
            raise RuntimeError(
                f"Nessun dato per {symbol}"
            )

        df = pd.DataFrame(values)

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
            df["datetime"],
            errors="coerce",
        )

        return (
            df.dropna(
                subset=["datetime", "close"]
            )
            .sort_values("datetime")[
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

    def _get_with_429_retry(
        self,
        url,
        params,
        timeout,
        max_attempts=3,
        wait_seconds=65,
    ):
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

                if response.status_code == 429:
                    if attempt < max_attempts - 1:
                        time.sleep(wait_seconds)
                        continue

                    response.raise_for_status()

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as error:
                last_error = error
                status = getattr(
                    error.response,
                    "status_code",
                    None,
                )

                if (
                    status == 429
                    and attempt < max_attempts - 1
                ):
                    time.sleep(wait_seconds)
                    continue

                raise

        if last_error:
            raise last_error

        raise RuntimeError(
            "Richiesta fallita senza risposta."
        )

    def daily_history(
        self,
        symbol,
        outputsize=300,
        adjust="splits",
    ):
        symbol = symbol.upper()

        key = (
            symbol,
            int(outputsize),
            str(adjust),
        )

        if key in self.cache:
            return self.cache[key].copy()

        disk = self._load_disk(
            symbol,
            outputsize,
            adjust,
        )

        if disk is not None:
            self.cache[key] = disk
            return disk.copy()

        response = self._get_with_429_retry(
            f"{self.BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": int(outputsize),
                "order": "ASC",
                "adjust": adjust,
                "apikey": self.api_key,
            },
            timeout=self.timeout,
        )

        df = self._parse_item(
            response.json(),
            symbol,
        )

        self.cache[key] = df

        self._save_disk(
            symbol,
            outputsize,
            adjust,
            df,
        )

        return df.copy()

    def batch_daily_history(
        self,
        symbols,
        outputsize=300,
        adjust="splits",
    ):
        symbols = list(
            dict.fromkeys(
                [
                    str(symbol).upper()
                    for symbol in symbols
                ]
            )
        )

        result = {}
        missing = []

        for symbol in symbols:
            key = (
                symbol,
                int(outputsize),
                str(adjust),
            )

            if key in self.cache:
                result[symbol] = (
                    self.cache[key].copy()
                )
                continue

            disk = self._load_disk(
                symbol,
                outputsize,
                adjust,
            )

            if disk is not None:
                self.cache[key] = disk
                result[symbol] = disk.copy()
            else:
                missing.append(symbol)

        chunks = [
            missing[
                index:index + self.batch_size
            ]
            for index in range(
                0,
                len(missing),
                self.batch_size,
            )
        ]

        for index, chunk in enumerate(chunks):
            if index > 0:
                time.sleep(61)

            try:
                response = self._get_with_429_retry(
                    f"{self.BASE}/time_series",
                    params={
                        "symbol": ",".join(chunk),
                        "interval": "1day",
                        "outputsize": int(outputsize),
                        "order": "ASC",
                        "adjust": adjust,
                        "apikey": self.api_key,
                    },
                    timeout=max(
                        self.timeout,
                        35,
                    ),
                    max_attempts=1,
                )

                data = response.json()

            except Exception:
                # Salta il pacchetto senza aggiungere
                # attese di diversi minuti.
                continue

            if (
                len(chunk) == 1
                and isinstance(data, dict)
                and "values" in data
            ):
                data = {
                    chunk[0]: data
                }

            if (
                isinstance(data, dict)
                and data.get("status") == "error"
            ):
                continue

            for symbol in chunk:
                try:
                    item = (
                        data.get(symbol, {})
                        if isinstance(data, dict)
                        else {}
                    )

                    df = self._parse_item(
                        item,
                        symbol,
                    )

                    key = (
                        symbol,
                        int(outputsize),
                        str(adjust),
                    )

                    self.cache[key] = df
                    result[symbol] = df.copy()

                    self._save_disk(
                        symbol,
                        outputsize,
                        adjust,
                        df,
                    )

                except Exception:
                    continue

        return result

    def press_releases(
        self,
        symbol,
        limit=8,
    ):
        response = self._get_with_429_retry(
            f"{self.BASE}/press_releases",
            params={
                "symbol": symbol,
                "apikey": self.api_key,
            },
            timeout=self.timeout,
            max_attempts=1,
        )

        data = response.json()

        if (
            isinstance(data, dict)
            and data.get("status") == "error"
        ):
            raise RuntimeError(
                data.get(
                    "message",
                    "Errore press releases Twelve Data",
                )
            )

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = (
                data.get("data")
                or data.get("press_releases")
                or data.get("values")
                or data.get("results")
                or []
            )
        else:
            items = []

        output = []

        for item in items[:int(limit)]:
            if isinstance(item, dict):
                output.append({
                    "title": (
                        item.get("title")
                        or item.get("headline")
                        or item.get("name")
                    ),
                    "text": (
                        item.get("body")
                        or item.get("text")
                        or item.get("summary")
                        or item.get("description")
                        or item.get("content")
                    ),
                    "datetime": (
                        item.get("datetime")
                        or item.get("published_at")
                        or item.get("date")
                        or item.get("timestamp")
                    ),
                    "url": (
                        item.get("url")
                        or item.get("link")
                    ),
                    "source": (
                        item.get("source")
                        or "Press release"
                    ),
                })

        return output
