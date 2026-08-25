# API e deploy

La documentazione corrente è in `README.md` e, nella cartella superiore, in
`LEGGIMI-PRIMA.md`, `SPECIFICA-COMPLETA.md` e
`IMPLEMENTATO-E-PREDISPOSTO.md`.

La produzione 2.1 usa solo dati reali. Il Dockerfile avvia:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Le chiavi EODHD/SEC restano sul server; l'APK contiene soltanto l'URL del
backend e la chiave applicativa inserita localmente dall'utente. Per conservare
la memoria del motore su Render è obbligatorio il disco `/var/data` descritto
nel file `LEGGIMI-PRIMA.md`.
