# Trading Agents Dashboard

MVP voor een multi-strategy trading research dashboard.

## Wat zit erin

- 6 strategie-agents: SMA trend, dual momentum, RSI mean reversion, breakout, low-volatility en defensive rotation.
- Uniforme backtest-engine met transactiekosten en slippage.
- KPI-scorebord: CAGR, max drawdown, Sharpe, Sortino, Calmar, win rate, profit factor, turnover/stability en samengestelde score.
- Streamlit-dashboard met overview, scoreboard, equity curves, drawdowns en actuele signalen.
- Fallback naar synthetische demo-data als yfinance niet beschikbaar is.

## Installatie

Optie A — met uv, werkt zonder aparte venv-setup:

```bash
cd /opt/data/home/trading-agents-dashboard
uv run --with pandas --with numpy --with plotly --with streamlit --with yfinance --with pytest pytest -q
```

Optie B — met eigen virtualenv, als `python3-venv` beschikbaar is:

```bash
cd /opt/data/home/trading-agents-dashboard
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```bash
PYTHONPATH=src uv run --with pytest --with pandas --with numpy pytest -q
```

## Dashboard starten

Met `uv`:

```bash
PYTHONPATH=src uv run --with pandas --with numpy --with plotly --with streamlit --with yfinance streamlit run app.py
```

Met standaard Python/pip:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```

Daarna open je de Streamlit URL in je browser, meestal `http://localhost:8501`.

Zie ook `DEPLOYMENT.md` voor Streamlit Cloud / VPS deployment.

Standaard tickers zijn Yahoo Finance proxies: SPY, QQQ, EFA, EEM, GLD, TLT, SHY. Voor EU/UCITS kun je later tickers aanpassen naar exchange-specifieke Yahoo-symbolen.

## Let op

Dit is research/backtesting-software, geen financieel advies en geen orderuitvoering. Backtests kunnen overfitten en houden beperkt rekening met echte liquiditeit, spreads, belasting en execution.
