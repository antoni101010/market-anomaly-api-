name: Verifica backend

on:
  push:
    branches: [main]
  workflow_dispatch: {}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v5

      - name: Installa Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'
          cache: pip

      - name: Installa dipendenze
        run: pip install -r requirements.txt

      - name: Compila sintassi Python
        run: python -m py_compile *.py providers/*.py api/*.py

      - name: Verifica funzioni critiche offline
        run: python release_verification.py

      - name: Esegue test backend
        run: pytest -q
