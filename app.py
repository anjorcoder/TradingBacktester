from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud runs app.py from the repository root and installs only
# requirements.txt. Add ./src explicitly so the local package imports work
# without requiring `pip install -e .` or PYTHONPATH configuration.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plotly.express as px
import streamlit as st

from trading_agents_dashboard.backtest import BacktestConfig, drawdown_frame, equity_frame, results_scoreboard, run_backtest
from trading_agents_dashboard.data import DEFAULT_TICKERS, load_prices
from trading_agents_dashboard.reporting import compact_scoreboard
from trading_agents_dashboard.strategies import default_agents

st.set_page_config(page_title="Trading Agents Dashboard", page_icon="📈", layout="wide")
st.title("📈 Trading Agents Strategy Dashboard")
st.caption("Research/backtesting MVP — geen financieel advies en geen orderuitvoering.")

with st.sidebar:
    st.header("Instellingen")
    tickers_text = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS))
    period = st.selectbox("Dataperiode", ["5y", "10y", "15y", "20y", "max"], index=1)
    initial_cash = st.number_input("Startkapitaal", value=10_000, min_value=1_000, step=1_000)
    transaction_cost_bps = st.number_input("Transactiekosten bps", value=2.0, min_value=0.0, step=0.5)
    slippage_bps = st.number_input("Slippage bps", value=3.0, min_value=0.0, step=0.5)
    run = st.button("Backtests draaien", type="primary")

tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
prices = load_prices(tickers, period=period)
config = BacktestConfig(initial_cash=initial_cash, transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps)
results = [run_backtest(agent, prices, config) for agent in default_agents()]
board = results_scoreboard(results)

leader = board.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Beste agent", leader["agent"], f"score {leader['score']:.1f}")
col2.metric("CAGR", f"{leader['cagr']:.2%}")
col3.metric("Max drawdown", f"{leader['max_drawdown']:.2%}")
col4.metric("Sharpe", f"{leader['sharpe']:.2f}")

st.subheader("Scorebord")
st.dataframe(
    compact_scoreboard(board),
    use_container_width=True,
    hide_index=True,
    column_config={
        "cagr": st.column_config.NumberColumn("CAGR", format="%.2f%%"),
        "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.2f%%"),
        "score": st.column_config.NumberColumn("Score", format="%.1f"),
    },
)

st.subheader("Equity curves")
eq = equity_frame(results)
st.plotly_chart(px.line(eq, x=eq.index, y=eq.columns, title="Equity curve per agent"), use_container_width=True)

st.subheader("Drawdowns")
dd = drawdown_frame(results)
st.plotly_chart(px.line(dd, x=dd.index, y=dd.columns, title="Drawdown per agent"), use_container_width=True)

st.subheader("Agent detail")
selected = st.selectbox("Kies agent", [r.agent_name for r in results])
result = next(r for r in results if r.agent_name == selected)
left, right = st.columns([1, 1])
with left:
    st.write(result.description)
    st.write("**Actueel signaal:**", result.current_signal)
    st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in result.kpis.items()})
with right:
    st.write("Laatste trades")
    if result.trades.empty:
        st.info("Geen trades in deze backtest.")
    else:
        st.dataframe(result.trades.tail(20), use_container_width=True, hide_index=True)

st.info(
    "Interpretatie: de score combineert rendement, drawdown, Sharpe/Sortino, Calmar, stabiliteit en turnover. "
    "Gebruik dit als researchlaag; valideer altijd out-of-sample en met realistische kosten/spreads."
)
