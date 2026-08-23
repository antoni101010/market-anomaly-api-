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

    def __init__(self, api_key, timeout=25, batch_size=8, cache_dir="data/price_cache"):
        self.api_key = api_key
        self.timeout = timeout
        # Piano gratuito Twelve Data: 8 crediti/minuto. Un pacchetto consuma
        # 1 credito PER SIMBOLO contenuto, quindi il pacchetto stesso non può
        # superare 8 simboli o sforiamo il limite già alla prima richiesta.
        self.batch_size = max(1, min(int(batch_size), 8))
        self.cache = {}
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=2.0, status_forcelist=[500,502,503,504], allowed_methods=["GET"])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _disk_key(self, symbol, outputsize, adjust):
        raw = f"{symbol.upper()}|{int(outputsize)}|{adjust}"
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def _disk_path(self, symbol, outputsize, adjust):
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{symbol.upper()}_{self._disk_key(symbol,outputsize,adjust)}.csv.gz"

    def _load_disk(self, symbol, outputsize, adjust):
        p = self._disk_path(symbol, outputsize, adjust)
        if not p or not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            return df.dropna(subset=["datetime","close"]).sort_values("datetime").reset_index(drop=True)
        except Exception:
            return None

    def _save_disk(self, symbol, outputsize, adjust, df):
        p = self._disk_path(symbol, outputsize, adjust)
        if p:
            try:
                df.to_csv(p, index=False, compression="gzip")
            except Exception:
                pass

    @staticmethod
    def _parse_item(item, symbol):
        if isinstance(item, dict) and item.get("status") == "error":
            raise RuntimeError(item.get("message", f"Errore Twelve Data per {symbol}"))
        values = item.get("values") if isinstance(item, dict) else None
        if not values:
            raise RuntimeError(f"Nessun dato per {symbol}")
        df = pd.DataFrame(values)
        if "volume" not in df.columns:
            df["volume"] = 0
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns:
                if c == "volume":
                    df[c] = 0
                else:
                    raise RuntimeError(f"Campo {c} mancante per {symbol}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.dropna(subset=["datetime","close"]).sort_values("datetime")[
            ["datetime","open","high","low","close","volume"]
        ].reset_index(drop=True)

    def daily_history(self, symbol, outputsize=300, adjust="splits"):
        symbol = symbol.upper()
        key = (symbol, int(outputsize), str(adjust))
        if key in self.cache:
            return self.cache[key].copy()

        disk = self._load_disk(symbol, outputsize, adjust)
        if disk is not None:
            self.cache[key] = disk
            return disk.copy()

        r = self.session.get(
            f"{self.BASE}/time_series",
            params={
                "symbol":symbol, "interval":"1day", "outputsize":int(outputsize),
                "order":"ASC", "adjust":adjust, "apikey":self.api_key,
            },
            timeout=self.timeout
        )
        r.raise_for_status()
        df = self._parse_item(r.json(), symbol)
        self.cache[key] = df
        self._save_disk(symbol, outputsize, adjust, df)
        return df.copy()

    def batch_daily_history(self, symbols, outputsize=300, adjust="splits"):
        symbols = list(dict.fromkeys([str(s).upper() for s in symbols]))
        result = {}
        missing = []

        for s in symbols:
            key = (s, int(outputsize), str(adjust))
            if key in self.cache:
                result[s] = self.cache[key].copy()
                continue
            disk = self._load_disk(s, outputsize, adjust)
            if disk is not None:
                self.cache[key] = disk
                result[s] = disk.copy()
            else:
                missing.append(s)

        # Pacchetti piccoli (<= 8 simboli, pari al limite di crediti/minuto) e
        # una pausa di un minuto pieno tra un pacchetto e l'altro: unico modo
        # per restare davvero dentro il piano gratuito senza errori 429.
        chunks = [missing[i:i+self.batch_size] for i in range(0, len(missing), self.batch_size)]
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(61)

            r = self.session.get(
                f"{self.BASE}/time_series",
                params={
                    "symbol":",".join(chunk), "interval":"1day",
                    "outputsize":int(outputsize), "order":"ASC",
                    "adjust":adjust, "apikey":self.api_key,
                },
                timeout=max(self.timeout, 35)
            )
            r.raise_for_status()
            data = r.json()

            if len(chunk) == 1 and isinstance(data, dict) and "values" in data:
                data = {chunk[0]: data}

            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(data.get("message", "Errore batch Twelve Data"))

            for sym in chunk:
                try:
                    item = data.get(sym, {}) if isinstance(data, dict) else {}
                    df = self._parse_item(item, sym)
                    key = (sym, int(outputsize), str(adjust))
                    self.cache[key] = df
                    result[sym] = df.copy()
                    self._save_disk(sym, outputsize, adjust, df)
                except Exception:
                    # A missing/delisted ticker must not kill an entire large-universe scan.
                    continue
