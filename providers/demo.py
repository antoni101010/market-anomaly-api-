import pandas as pd
import numpy as np

class DemoProvider:
    def __init__(self):
        self.cache = {}

    def daily_history(self, symbol, outputsize=300, adjust="all"):
        key = (symbol.upper(), int(outputsize))
        if key in self.cache:
            return self.cache[key].copy()

        sym = symbol.upper()
        seed = sum(ord(c) for c in sym)
        rng = np.random.default_rng(seed)
        n = max(1700, int(outputsize))
        is_benchmark = sym in {"SPY","QQQ","XLK","XLY","XLC","XLF","XLV","XLI"}

        drift = rng.uniform(0.00018, 0.00055) if is_benchmark else rng.uniform(0.00005, 0.00065)
        vol = rng.uniform(0.008, 0.013) if is_benchmark else rng.uniform(0.012, 0.024)
        rets = rng.normal(drift, vol, n)

        if not is_benchmark:
            # Eventi sintetici distribuiti nello storico. Alcuni recuperano,
            # altri no: il backtest non deve essere perfetto.
            event_points = list(range(330 + seed % 35, n - 130, 180 + seed % 23))
            for j, ep in enumerate(event_points):
                shock = 0.06 + ((seed + j) % 9) / 100
                rets[ep] -= shock
                if (seed + j) % 4 != 0:
                    # recupero graduale nelle settimane successive
                    rebound = shock * rng.uniform(0.45, 0.85)
                    rets[ep+3:ep+28] += rebound / 25.0
                else:
                    # caso value-trap-like: debolezza che persiste
                    rets[ep+3:ep+45] -= rng.uniform(0.0004, 0.0012)

        close = 100 * np.cumprod(1 + rets)
        open_ = close * (1 + rng.normal(0, 0.0025, n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.010, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.010, n))
        volume = rng.integers(700_000, 4_000_000, n).astype(float)
        if not is_benchmark:
            for ep in range(330 + seed % 35, n - 2, 180 + seed % 23):
                volume[ep] *= rng.uniform(2.0, 4.0)

        # Nota: su alcune versioni di pandas, date_range(periods=n, freq="B") può
        # restituire n-1 date per via di come viene calcolato l'offset business-day.
        # Generiamo qualche data in più e tagliamo esattamente a n per essere sicuri
        # che l'array di date abbia sempre la stessa lunghezza degli array di prezzo.
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 5)[-n:]

        df = pd.DataFrame({
            "datetime": dates,
            "open": open_, "high": high, "low": low, "close": close, "volume": volume
        }).tail(min(int(outputsize), n)).reset_index(drop=True)
        self.cache[key] = df
        return df.copy()

    def batch_daily_history(self, symbols, outputsize=300, adjust="all"):
        return {s: self.daily_history(s, outputsize) for s in symbols}

    def press_releases(self, symbol, limit=8):
        seed = sum(ord(c) for c in symbol.upper())
        samples = {
            0: [
                {"title": f"{symbol} reports quarterly results and reiterates guidance",
                 "text": "Revenue growth remains positive. Management reiterates full-year outlook.",
                 "source": "Demo press release"},
                {"title": f"{symbol} announces share repurchase authorization",
                 "text": "Board approves a new share repurchase program.",
                 "source": "Demo press release"},
            ],
            1: [
                {"title": f"{symbol} lowers guidance after weak demand",
                 "text": "Company cuts guidance after a temporary slowdown and inventory normalization.",
                 "source": "Demo press release"},
            ],
            2: [
                {"title": f"{symbol} announces quarterly results",
                 "text": "Revenue misses estimates while margins remain stable. Outlook unchanged.",
                 "source": "Demo press release"},
            ],
            3: [
                {"title": f"{symbol} discloses regulatory investigation",
                 "text": "Company received a subpoena related to an ongoing investigation.",
                 "source": "Demo press release"},
            ],
        }
        return samples[seed % 4][:int(limit)]
