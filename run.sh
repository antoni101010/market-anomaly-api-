#!/usr/bin/env sh
set -eu

# Avvio di produzione: la release 2.0 espone esclusivamente l'API real-data.
# PORT viene fornita da Render; il valore 8000 permette anche l'avvio locale.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
