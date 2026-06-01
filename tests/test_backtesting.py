import numpy as np
import pandas as pd

from trading_agents_dashboard.backtest import BacktestConfig, run_backtest
from trading_agents_dashboard.kpis import calculate_kpis, score_strategy
from trading_agents_dashboard.strategies import (
    BreakoutAgent,
    DefensiveRotationAgent,
    DualMomentumAgent,
    LowVolatilityAgent,
    RSIMeanReversionAgent,
    SMATrendAgent,
)


def make_prices(rows=260):
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    base = np.linspace(100, 160, rows)
    prices = pd.DataFrame(
        {
            "AAA": base,
            "BBB": np.linspace(100, 90, rows),
            "CCC": 100 + 10 * np.sin(np.linspace(0, 12, rows)),
        },
        index=dates,
    )
    return prices


def test_kpis_include_core_metrics_and_reasonable_values():
    equity = pd.Series([100, 102, 101, 106, 110], index=pd.date_range("2020-01-01", periods=5, freq="B"))
    trades = pd.DataFrame({"asset": ["AAA"], "entry_date": [equity.index[0]], "exit_date": [equity.index[-1]], "return": [0.10]})

    kpis = calculate_kpis(equity, trades, periods_per_year=252)

    assert set(["cagr", "max_drawdown", "sharpe", "sortino", "calmar", "win_rate", "profit_factor", "total_return"]).issubset(kpis)
    assert kpis["total_return"] == 0.10
    assert kpis["max_drawdown"] < 0
    assert kpis["win_rate"] == 1.0


def test_score_strategy_penalizes_high_drawdown_and_turnover():
    strong = {"cagr": 0.18, "max_drawdown": -0.10, "sharpe": 1.4, "sortino": 1.8, "calmar": 1.8, "trades_per_year": 6, "stability": 0.8}
    weak = {"cagr": 0.18, "max_drawdown": -0.45, "sharpe": 0.3, "sortino": 0.4, "calmar": 0.4, "trades_per_year": 180, "stability": 0.2}

    assert score_strategy(strong) > score_strategy(weak)
    assert 0 <= score_strategy(strong) <= 100


def test_backtest_returns_equity_trades_positions_and_costs():
    prices = make_prices()
    agent = SMATrendAgent(window=50)

    result = run_backtest(agent, prices, BacktestConfig(initial_cash=10_000, transaction_cost_bps=5, slippage_bps=5))

    assert result.agent_name == "SMA Trend Agent"
    assert result.equity.iloc[0] == 10_000
    assert len(result.equity) == len(prices)
    assert set(result.positions.columns) == set(prices.columns)
    assert set(result.target_weights.index) == set(prices.columns)
    assert result.target_weights.sum() <= 1.000001
    assert "cagr" in result.kpis
    assert result.kpis["costs_paid"] >= 0


def test_all_default_agents_generate_valid_weight_frames():
    prices = make_prices(320)
    agents = [
        SMATrendAgent(window=80),
        DualMomentumAgent(lookback=60, top_n=1),
        RSIMeanReversionAgent(rsi_window=14),
        BreakoutAgent(lookback=55),
        LowVolatilityAgent(vol_window=40, top_n=1),
        DefensiveRotationAgent(momentum_lookback=60),
    ]

    for agent in agents:
        weights = agent.generate_weights(prices)
        assert weights.index.equals(prices.index)
        assert set(weights.columns) == set(prices.columns)
        assert (weights.sum(axis=1) <= 1.000001).all()
        assert (weights.fillna(0) >= 0).all().all()
