from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kpis import calculate_kpis, drawdown_series, score_strategy
from .strategies import StrategyAgent


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 3.0
    periods_per_year: int = 252


@dataclass
class BacktestResult:
    agent_name: str
    description: str
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    target_weights: pd.Series
    trades: pd.DataFrame
    kpis: dict[str, float]
    current_signal: str


def _trade_log(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    active: dict[str, tuple[pd.Timestamp, float]] = {}
    for date in weights.index:
        for asset in weights.columns:
            weight = float(weights.loc[date, asset])
            price = float(prices.loc[date, asset]) if pd.notna(prices.loc[date, asset]) else np.nan
            if weight > 0 and asset not in active and np.isfinite(price):
                active[asset] = (date, price)
            elif weight == 0 and asset in active and np.isfinite(price):
                entry_date, entry_price = active.pop(asset)
                rows.append({
                    "asset": asset,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "return": price / entry_price - 1.0,
                })
    last_date = weights.index[-1]
    for asset, (entry_date, entry_price) in active.items():
        price = float(prices.loc[last_date, asset])
        rows.append({
            "asset": asset,
            "entry_date": entry_date,
            "exit_date": last_date,
            "entry_price": entry_price,
            "exit_price": price,
            "return": price / entry_price - 1.0,
        })
    return pd.DataFrame(rows)


def run_backtest(agent: StrategyAgent, prices: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    if prices.empty:
        raise ValueError("prices cannot be empty")
    prices = prices.sort_index().ffill().dropna(how="all")
    prices = prices.dropna(axis=1, how="any")
    if prices.empty:
        raise ValueError("prices must contain at least one complete price column")

    raw_weights = agent.generate_weights(prices).reindex_like(prices).fillna(0.0)
    raw_weights = raw_weights.div(raw_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Shift by one bar: today's signal is executed for tomorrow's return, limiting look-ahead bias.
    positions = raw_weights.shift(1).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    gross_returns = (positions * asset_returns).sum(axis=1)

    turnover = raw_weights.diff().abs().sum(axis=1).fillna(raw_weights.abs().sum(axis=1))
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    costs = turnover * cost_rate
    net_returns = gross_returns - costs

    equity = config.initial_cash * (1 + net_returns).cumprod()
    equity.iloc[0] = config.initial_cash
    trades = _trade_log(raw_weights, prices)
    kpis = calculate_kpis(equity, trades, config.periods_per_year)
    years = max(len(equity) / config.periods_per_year, 1 / config.periods_per_year)
    kpis["trades"] = float(len(trades))
    kpis["trades_per_year"] = float(len(trades) / years)
    kpis["avg_exposure"] = float(positions.sum(axis=1).mean())
    kpis["costs_paid"] = float((costs * equity.shift(1).fillna(config.initial_cash)).sum())
    kpis["score"] = score_strategy(kpis)

    return BacktestResult(
        agent_name=agent.name,
        description=agent.description,
        equity=equity.rename(agent.name),
        returns=net_returns.rename(agent.name),
        positions=positions,
        target_weights=raw_weights.iloc[-1].rename("target_weight"),
        trades=trades,
        kpis=kpis,
        current_signal=agent.current_signal(prices),
    )


def results_scoreboard(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"agent": result.agent_name, "description": result.description, "current_signal": result.current_signal}
        row.update(result.kpis)
        rows.append(row)
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values("score", ascending=False).reset_index(drop=True)
        board.insert(0, "rank", range(1, len(board) + 1))
    return board


def equity_frame(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.concat([r.equity for r in results], axis=1)


def drawdown_frame(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.concat([drawdown_series(r.equity).rename(r.agent_name) for r in results], axis=1)
