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
from trading_agents_dashboard.execution import (
    ReadinessRules,
    assess_readiness,
    build_position_plan,
    parse_holdings_text,
)
from trading_agents_dashboard.reporting import compact_scoreboard
from trading_agents_dashboard.strategies import default_agents


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers: tuple[str, ...], period: str):
    return load_prices(list(tickers), period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_backtests(
    tickers: tuple[str, ...],
    period: str,
    initial_cash: float,
    transaction_cost_bps: float,
    slippage_bps: float,
):
    prices = cached_prices(tickers, period)
    config = BacktestConfig(
        initial_cash=initial_cash,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    results = [run_backtest(agent, prices, config) for agent in default_agents()]
    board = results_scoreboard(results)
    return prices, results, board


st.set_page_config(page_title="Trading Agents Dashboard", page_icon="📈", layout="wide")
st.title("📈 Trading Agents Strategy Dashboard")
st.caption("Research/backtesting MVP — geen financieel advies en geen orderuitvoering.")

with st.sidebar:
    st.header("Instellingen")
    with st.form("backtest_settings"):
        tickers_text = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS))
        period = st.selectbox("Dataperiode", ["5y", "10y", "15y", "20y", "max"], index=1)
        initial_cash = st.number_input("Startkapitaal", value=10_000, min_value=1_000, step=1_000)
        transaction_cost_bps = st.number_input("Transactiekosten bps", value=2.0, min_value=0.0, step=0.5)
        slippage_bps = st.number_input("Slippage bps", value=3.0, min_value=0.0, step=0.5)
        run = st.form_submit_button("Backtests draaien", type="primary")

    st.caption("Wijzig instellingen en klik op de knop om opnieuw te berekenen.")

if not tickers_text.strip():
    st.error("Vul minimaal één ticker in, bijvoorbeeld SPY, QQQ, GLD.")
    st.stop()

tickers = tuple(ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip())
params = (tickers, period, float(initial_cash), float(transaction_cost_bps), float(slippage_bps))

if run or "backtest_payload" not in st.session_state or st.session_state.get("backtest_params") != params:
    with st.spinner("Data ophalen en agents backtesten..."):
        try:
            st.session_state["backtest_payload"] = cached_backtests(*params)
            st.session_state["backtest_params"] = params
        except Exception as exc:
            st.error(f"Backtest kon niet worden uitgevoerd: {exc}")
            st.stop()

prices, results, board = st.session_state["backtest_payload"]

if board.empty or not results:
    st.error("Geen backtestresultaten beschikbaar. Controleer de tickers of probeer een kortere periode.")
    st.stop()

st.success(f"Backtest klaar: {len(results)} agents, {len(prices)} koersdagen, {len(prices.columns)} assets.")

leader = board.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Beste agent", leader["agent"], f"score {leader['score']:.1f}")
col2.metric("CAGR", f"{leader['cagr']:.2%}")
col3.metric("Max drawdown", f"{leader['max_drawdown']:.2%}")
col4.metric("Sharpe", f"{leader['sharpe']:.2f}")

st.subheader("Scorebord")
st.dataframe(
    compact_scoreboard(board),
    width="stretch",
    hide_index=True,
    column_config={
        "cagr": st.column_config.NumberColumn("CAGR", format="%.4f"),
        "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.4f"),
        "score": st.column_config.NumberColumn("Score", format="%.1f"),
    },
)

st.subheader("Equity curves")
eq = equity_frame(results)
st.plotly_chart(px.line(eq, x=eq.index, y=eq.columns, title="Equity curve per agent"), width="stretch")

st.subheader("Drawdowns")
dd = drawdown_frame(results)
st.plotly_chart(px.line(dd, x=dd.index, y=dd.columns, title="Drawdown per agent"), width="stretch")

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
        st.dataframe(result.trades.tail(20), width="stretch", hide_index=True)

st.subheader("Strategie in werking zetten")
st.caption(
    "Deze sectie vertaalt de gekozen strategie naar een paper-trading/rebalance-plan. "
    "Het voert géén orders uit. Gebruik dit als checklist voordat je iets live toepast."
)

rules_col, plan_col = st.columns([1, 1])
with rules_col:
    st.markdown("**1. Readiness rules**")
    min_score = st.slider("Minimum strategy score", 0, 100, 65)
    max_dd = st.slider("Max toegestane drawdown", 0.05, 0.80, 0.30, 0.05, format="%.0f%%")
    max_trades = st.slider("Max trades per jaar", 1, 200, 36)
    min_calmar = st.slider("Minimum Calmar", 0.0, 3.0, 0.5, 0.1)
    assessment = assess_readiness(
        result.kpis,
        ReadinessRules(
            min_score=float(min_score),
            max_drawdown=float(max_dd),
            max_trades_per_year=float(max_trades),
            min_calmar=float(min_calmar),
        ),
    )
    if assessment.status == "READY":
        st.success("Status: READY voor paper/live follow-up volgens deze regels")
    elif assessment.status == "PAPERTRADE":
        st.warning("Status: PAPERTRADE — eerst volgen zonder live geld")
    else:
        st.error("Status: REJECT — niet robuust genoeg volgens deze regels")

    if assessment.failed_checks:
        st.write("**Gefaalde checks:**")
        for check in assessment.failed_checks:
            st.write(f"- {check}")
    if assessment.warnings:
        st.write("**Waarschuwingen:**")
        for warning in assessment.warnings:
            st.write(f"- {warning}")

with plan_col:
    st.markdown("**2. Rebalance-plan**")
    paper_value = st.number_input("Paper/live portefeuillewaarde", value=10_000, min_value=100, step=500)
    min_trade_value = st.number_input("Negeer trades kleiner dan", value=25, min_value=0, step=5)
    holdings_text = st.text_area(
        "Huidige holdings optioneel, formaat: SPY: 3",
        value="",
        placeholder="SPY: 3\nQQQ: 1.5\nGLD: 2",
    )
    latest_prices = prices.iloc[-1]
    current_holdings = parse_holdings_text(holdings_text)
    plan = build_position_plan(
        prices=latest_prices,
        target_weights=result.target_weights,
        portfolio_value=float(paper_value),
        current_holdings=current_holdings,
        min_trade_value=float(min_trade_value),
    )
    st.dataframe(
        plan,
        width="stretch",
        hide_index=True,
        column_config={
            "target_weight": st.column_config.NumberColumn("Target gewicht", format="%.2f"),
            "price": st.column_config.NumberColumn("Laatste prijs", format="%.2f"),
            "target_value": st.column_config.NumberColumn("Doelwaarde", format="%.2f"),
            "current_value": st.column_config.NumberColumn("Huidige waarde", format="%.2f"),
            "trade_value": st.column_config.NumberColumn("Trade waarde", format="%.2f"),
            "trade_shares": st.column_config.NumberColumn("Aantal shares", format="%.4f"),
        },
    )

st.markdown("**3. Praktische checklist vóór live toepassen**")
st.write(
    "- Start met minimaal 1–3 maanden paper trading.\n"
    "- Rebalance op vaste momenten, bijvoorbeeld wekelijks of maandelijks; niet impulsief intraday.\n"
    "- Check of de gebruikte tickers echt bij je broker beschikbaar zijn en voldoende liquide zijn.\n"
    "- Gebruik realistische kosten/slippage en vergelijk altijd met buy-and-hold benchmark.\n"
    "- Bepaal vooraf maximale positieomvang, maximale drawdown waarbij je pauzeert, en evaluatiemomenten."
)

st.info(
    "Interpretatie: de score combineert rendement, drawdown, Sharpe/Sortino, Calmar, stabiliteit en turnover. "
    "Gebruik dit als researchlaag; valideer altijd out-of-sample en met realistische kosten/spreads."
)
