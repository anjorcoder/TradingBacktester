# Browser deployment

## Optie 1: lokaal in je browser

```bash
cd trading-agents-dashboard
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```

Open daarna de URL die Streamlit toont, meestal:

- http://localhost:8501

## Optie 2: Streamlit Community Cloud

1. Push deze repo naar GitHub.
2. Ga naar https://share.streamlit.io/
3. Kies **New app**.
4. Selecteer repository `trading-agents-dashboard`.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Deploy.

Streamlit Cloud gebruikt `requirements.txt` voor dependencies.

## Optie 3: server/VPS

```bash
cd trading-agents-dashboard
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open firewall/security group voor poort `8501`, of zet Nginx/Caddy ervoor met HTTPS.

## Let op

Dit dashboard is research/backtesting-software. Het voert geen orders uit en is geen financieel advies.
