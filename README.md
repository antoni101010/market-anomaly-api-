# Market Anomaly API 2.1

Backend FastAPI real-data-only per l'app Android Market Anomaly.

## Avvio

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Copia i valori di `.env.example` nelle variabili d'ambiente del server. Sono
obbligatorie `MARKET_ANOMALY_DATA_MODE=live`, `EODHD_API_KEY`,
`MARKET_ANOMALY_API_KEY` e un `SEC_USER_AGENT` identificabile.

Su Render monta un disco in `/var/data` e usa:

```text
MARKET_ANOMALY_DB=/var/data/market_anomaly.db
MARKET_ANOMALY_PRICE_CACHE_DIR=/var/data/price_cache
MARKET_ANOMALY_BACKUP_DIR=/var/data/backups
```

## Endpoint principali

| Metodo | Endpoint | Funzione |
|---|---|---|
| GET | `/health` | versione e stato server |
| POST | `/api/scan` | avvia Light + Deep scanner |
| GET | `/api/scan/status` | stato scansione background |
| GET | `/api/dashboard` | risultati con filtri personali |
| GET | `/api/ticker/{ticker}` | dettaglio, prezzo e narrativa |
| GET | `/api/ticker/{ticker}/prices` | grafico multi-periodo |
| GET | `/api/search` | ricerca globale provider |
| POST | `/api/analyze` | analisi manuale titolo |
| GET/POST/DELETE | `/api/watchlist` | titoli seguiti |
| GET | `/api/history` | storico segnali |
| POST | `/api/feedback` | utile / possibile falso segnale |
| GET | `/api/learning` | esiti aggregati per orizzonte |
| POST | `/api/outcomes/update` | aggiorna esiti maturati |
| GET | `/api/diagnostics` | prezzi, storage e scanner |

Tranne `/health`, gli endpoint richiedono l'header `X-API-Key` quando
`MARKET_ANOMALY_API_KEY` è configurata.

## Test

```bash
python -m py_compile *.py providers/*.py api/*.py
python release_verification.py
pytest -q
```

Il workflow GitHub `Verifica backend` esegue gli stessi controlli a ogni push.
I file demo/backtest restano esclusivamente come dataset di ricerca offline;
l'API di produzione rifiuta `MARKET_ANOMALY_DATA_MODE` diverso da `live`.
