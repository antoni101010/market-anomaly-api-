# Market Anomaly API — avvio e deploy

## Cosa è cambiato rispetto alla v1.0

- **Bug corretto** in `storage.py` (mancava `import math`, `save_signals()` andava in errore).
- **Bug corretto** in `providers/demo.py` (generazione date incoerente su pandas recenti).
- Rimosso `signal_store.py` (file morto, non più usato).
- Aggiunte a `storage.py` le tabelle `watchlist` e `latest_scan` (necessarie perché un'API
  stateless non può usare `st.session_state` di Streamlit come faceva la UI originale).
- Nuovo `narrative.py`: genera le sezioni "Perché potrebbe essere un'anomalia" /
  "Perché potrebbe non esserlo" e la classificazione in fasce (soglie configurabili).
- Nuovo `service.py`: incapsula scanner/model/catalyst in funzioni pulite, riusate sia
  dall'API sia (in futuro) da uno scheduler.
- Nuovo `api/main.py`: backend **FastAPI** con gli endpoint per l'app Android.
- `app.py` (Streamlit) **resta invariato** e continua a funzionare per uso interno/debug
  con `streamlit run app.py` — è il tuo pannello di analisi avanzata (backtest, statistica,
  walk-forward, bias audit), che l'app Android non replica nella v1.

## Avvio in locale (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# apri .env e imposta almeno MARKET_ANOMALY_API_KEY

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Testa che risponda (da un altro terminale o dal browser):
```
http://localhost:8000/health
```

Poi, con l'header `X-API-Key` impostato al valore scelto in `.env`:
```
POST http://localhost:8000/api/scan?limit=50
GET  http://localhost:8000/api/dashboard
GET  http://localhost:8000/api/ticker/H003
GET  http://localhost:8000/api/watchlist
GET  http://localhost:8000/api/history
```

In modalità demo (`MARKET_ANOMALY_DATA_MODE=demo`, il default) non serve nessuna API key
di dati finanziari: tutto funziona con dati sintetici, gratis, subito.

## Endpoint disponibili

| Metodo | Path | Descrizione |
|---|---|---|
| GET | `/health` | Stato del server (nessuna autenticazione) |
| POST | `/api/scan` | Esegue una nuova scansione e la salva |
| GET | `/api/dashboard` | Classifica anomalie dell'ultima scansione |
| GET | `/api/ticker/{ticker}` | Dettaglio completo di un titolo |
| GET | `/api/watchlist` | Lista titoli seguiti con performance |
| POST | `/api/watchlist` | Aggiunge un ticker (body: `{"ticker":"AAPL"}`) |
| DELETE | `/api/watchlist/{ticker}` | Rimuove un ticker |
| GET | `/api/history` | Storico segnali salvati nel tempo |

## Deploy su Railway (consigliato per iniziare)

1. Vai su [railway.app](https://railway.app) e crea un account (gratis, no carta di credito
   necessaria per il piano di prova).
2. "New Project" → "Deploy from GitHub repo" → seleziona il repository dove hai caricato
   questo codice (ricorda: deve includere anche la cartella `api/`).
3. Railway rileva il `Dockerfile` e builda automaticamente l'immagine.
4. Vai in **Variables** e imposta almeno:
   - `MARKET_ANOMALY_API_KEY` = una chiave a tua scelta (usala poi anche nell'app Android)
   - `MARKET_ANOMALY_DATA_MODE` = `demo` (per iniziare) oppure `live` se hai già una chiave
     Twelve Data
   - se `live`: `TWELVE_DATA_API_KEY`, `SEC_USER_AGENT`
5. Vai in **Settings → Networking** e genera un dominio pubblico (es.
   `market-anomaly-production.up.railway.app`).
6. **Importante — persistenza dati**: in Settings aggiungi un **Volume** montato su
   `/app/data`, altrimenti watchlist e storico si perdono a ogni riavvio del container.
7. Testa `https://<il-tuo-dominio>/health` dal browser.

Questo URL pubblico sarà quello che l'app Android chiamerà come base URL dell'API.

## Nota sulle API key

Le chiavi di Twelve Data / Alpha Vantage vivono **solo** come variabili d'ambiente sul
server Railway. L'app Android non le contiene mai: parla solo con la tua API, che parla
lei con i provider di dati. Questo è già coerente con quanto richiesto nel brief
originale (nessuna API key privata dentro l'APK).
