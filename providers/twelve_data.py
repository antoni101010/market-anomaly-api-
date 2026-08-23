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
        # Nessun retry automatico su 429 qui: lo gestiamo noi manualmente,
        # con attese complete di un minuto, per rispettare davvero il reset
        # del contatore di crediti (un retry troppo rapido peggiora le cose).
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

    def _get_with_429_retry(self, url, params, timeout, max_attempts=3, wait_seconds=65):
        """GET resiliente al limite di crediti/minuto: su 429 aspetta un minuto
        pieno e riprova, invece di arrendersi subito o ritentare troppo in fretta."""
        last_exc = None
        for attempt in range(max_attempts):
            try:
                r = self.session.get(url, params=params, timeout=timeout)
                if r.status_code == 429:
                    if attempt < max_attempts - 1:
                        time.sleep(wait_seconds)
                        continue
                    r.raise_for_status()
                r.raise_for_status()
                return r
            except requests.exceptions.HTTPError as e:
                last_exc = e
                if getattr(e.response, "status_code", None) == 429 and attempt < max_attempts - 1:
                    time.sleep(wait_seconds)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Richiesta fallita senza risposta.")

    def daily_history(self, symbol, outputsize=300, adjust="splits"):
        symbol = symbol.upper()
        key = (symbol, int(outputsize), str(adjust))
        if key in self.cache:
            return self.cache[key].copy()

        disk = self._load_disk(symbol, outputsize, adjust)
        if disk is not None:
            self.cache[key] = disk
            return disk.copy()

        r = self._get_with_429_retry(
            f"{self.BASE}/time_series",
            params={
                "symbol":symbol, "interval":"1day", "outputsize":int(outputsize),
                "order":"ASC", "adjust":adjust, "apikey":self.api_key,
            },
            timeout=self.timeout
        )
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
        # una pausa di un minuto pieno tra un pacchetto e l'altro. Se un
        # pacchetto fallisce comunque per limite crediti, lo si ritenta invece
        # di abortire l'intera scansione (che butterebbe via anche i dati già
        # ottenuti per i pacchetti precedenti).
        chunks = [missing[i:i+self.batch_size] for i in range(0, len(missing), self.batch_size)]
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(61)

            try:
                r = self._get_with_429_retry(
                    f"{self.BASE}/time_series",
                    params={
                        "symbol":",".join(chunk), "interval":"1day",
                        "outputsize":int(outputsize), "order":"ASC",
                        "adjust":adjust, "apikey":self.api_key,
                    },
                    timeout=max(self.timeout, 35)
                )
                data = r.json()
            except Exception:
                # Questo pacchetto non ce l'ha fatta nemmeno dopo i tentativi:
                # saltalo e prosegui con i successivi, non perdere tutto il resto.
                continue

            if len(chunk) == 1 and isinstance(data, dict) and "values" in data:
                data = {chunk[0]: data}

            if isinstance(data, dict) and data.get("status") == "error":
                continue

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

        return result

    def press_releases(self, symbol, limit=8):
        r = self._get_with_429_retry(
            f"{self.BASE}/press_releases",
            params={"symbol":symbol, "apikey":self.api_key},
            timeout=self.timeout
        )
        data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(data.get("message", "Errore press releases Twelve Data"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("data") or data.get("press_releases") or data.get("values") or data.get("results") or []
        else:
            items = []
        out = []
        for x in items[:int(limit)]:
            if isinstance(x, dict):
                out.append({
                    "title":x.get("title") or x.get("headline") or x.get("name"),
                    "text":x.get("body") or x.get("text") or x.get("summary") or x.get("description") or x.get("content"),
                    "datetime":x.get("datetime") or x.get("published_at") or x.get("date") or x.get("timestamp"),
                    "url":x.get("url") or x.get("link"),
                    "source":x.get("source") or "Press release",
                })
        return out
