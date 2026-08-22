# Market Anomaly v1.0

Versione completa del motore di ricerca delle anomalie di mercato.

## Funzioni incluse

### Live scanner
- Anomaly Score 0–100
- Opportunity Score 0–100
- Recovery Potential
- Quality Score
- Value Trap Risk
- Catalyst Risk
- confronto con SPY e settore
- drawdown, RSI, volume relativo, momentum e shock

### Catalyst engine
Analizza comunicati societari e filing SEC recenti per distinguere segnali potenzialmente temporanei da rischi strutturali.

### Fondamentali
SEC EDGAR per società USA, con modalità point-in-time nel backtest.

### Backtest serio
- universo storico dinamico
- titoli delistati
- terminal delisting return quando disponibile
- prezzi adjusted
- commissioni e slippage
- top-N + cooldown
- confronto con SPY

### Statistica
- bootstrap dell'excess return
- intervallo di confidenza 95%
- sign-flip permutation test
- analisi per settore
- analisi bull/bear/neutral
- correzione FDR Benjamini-Hochberg
- curva cohort e max drawdown

### Robustezza
- train/holdout
- walk-forward
- stabilità dei pesi
- Bias Audit A–E

### Prodotto
- storico segnali in SQLite
- regole alert
- export CSV
- Dockerfile
- cache dati
- configurazione tramite variabili ambiente

## Avvio rapido

```bash
pip install -r requirements.txt
streamlit run app.py
```

Per provare tutto senza costi seleziona **Demo**.

## Dati reali

### Twelve Data
Usato per serie storiche/live e comunicati societari. L'app usa batch e cache locale.

### SEC EDGAR
Usato per fondamentali e filing. Imposta:

```bash
export SEC_USER_AGENT="MarketAnomaly tua-email@example.com"
```

### Alpha Vantage
Opzionale per metadata di listing/delisting dell'universo storico.

## Nota commerciale
La v1.0 è il motore applicativo completo, ma per vendere il servizio servono comunque contratti dati, hosting, login, pagamenti e revisione legale/compliance. Questi sono servizi esterni al motore di analisi.

Vedi `COMMERCIAL_CHECKLIST.md`.
