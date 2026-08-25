# Market Anomaly API 2.3

Backend FastAPI real-data-only per Market Anomaly.

## Scanner globale v2.3

La scansione di produzione non parte più da poche decine di ribassi giornalieri.
Con EODHD usa il Bulk EOD Extended sugli exchange globali configurati e passa
**ogni azione ordinaria eleggibile** attraverso il Light Scanner. Il limite di
sicurezza predefinito è 50.000 righe, quindi non tronca artificialmente un
universo da 10.000/20.000+ titoli.

Sono esclusi dal main scanner gli strumenti non coerenti con la specifica
(penny/illiquidi, fondi/ETF, warrant, preferred, bond e capitalizzazioni sotto
la soglia core). Il Light ranking usa drawdown 250d, shock giornaliero, volume
anomalo e distanza dalle EMA; il Deep Engine approfondisce poi fino a 300
candidati globali, con copertura geografica, fondamentali e fino a 120 analisi
di news/catalizzatori.

La dashboard distingue:

- universo globale scansionato;
- candidati Light statisticamente rilevanti;
- analisi Deep completate;
- risultati mostrati dai filtri.

I fondamentali e lo storico vengono memorizzati in cache per ridurre chiamate
ripetitive. Il workflow orario usa il backend centrale; l'app mobile non
contiene chiavi del provider.

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

Su Render con disco persistente:

```text
MARKET_ANOMALY_DB=/var/data/market_anomaly.db
MARKET_ANOMALY_PRICE_CACHE_DIR=/var/data/price_cache
MARKET_ANOMALY_BACKUP_DIR=/var/data/backups
```

La v2.3 mantiene comunque un core globale di exchange e limiti minimi moderni
anche se su Render sono rimaste vecchie variabili della v2.2.

## Endpoint principali

| Metodo | Endpoint | Funzione |
|---|---|---|
| GET | `/health` | versione e stato server |
| POST | `/api/scan` | avvia Light globale + Deep scanner |
| GET | `/api/scan/status` | stato scansione background |
| GET | `/api/dashboard` | ranking e filtri |
| GET | `/api/market-tension` | tensione globale e diagnostica |
| GET | `/api/ticker/{ticker}` | dettaglio statistico |
| GET | `/api/ticker/{ticker}/prices` | 1G/5G/1M/6M/1A/5A |
| GET | `/api/search` | ricerca globale provider |
| POST | `/api/analyze` | analisi manuale titolo |
| GET/POST/DELETE | `/api/watchlist` | titoli seguiti |
| GET | `/api/history` | storico segnali |
| POST | `/api/feedback` | feedback sul modello |
| GET | `/api/learning` | esiti aggregati |
| POST | `/api/outcomes/update` | aggiorna esiti maturati |
| GET | `/api/diagnostics` | storage/scanner/provider |

## Correzioni dati v2.3

- periodi grafico basati su calendario reale;
- min/max/drawdown calcolati prima del downsampling;
- intraday 1G/5G sanitizzato da `NaN/null`, con fallback giornaliero esplicito;
- prezzo/ricavi ricavato anche da market cap / ricavi TTM quando necessario;
- cash runway positivo = non applicabile, non dato mancante;
- Confidence combina fondamentali, freschezza prezzo, copertura valutativa e
  catalizzatore, quindi non mostra più 100/100 con livelli non analizzati;
- news EODHD integrate nel Catalyst Engine;
- tensione globale su campione neutrale multi-mercato e diagnostica visibile.

## Test

```bash
python -m py_compile *.py providers/*.py api/*.py
python release_verification.py
pytest -q
```

La release v2.3 include 28 test backend. I test di regressione coprono anche
APP/AppLovin, selezione di 300 Deep su un universo sintetico da 20.000 titoli,
riconciliazione NVDA/listing primaria, finestre prezzo calendar-based e dati
intraday con campi null/NaN.
