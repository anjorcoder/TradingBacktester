from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_float(value: float, default: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def calculate_kpis(equity: pd.Series, trades: pd.DataFrame | None = None, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate core performance KPIs from an equity curve and optional trade log."""
    clean = equity.dropna().astype(float)
    if clean.empty or clean.iloc[0] <= 0:
        raise ValueError("equity must contain positive values")

    returns = clean.pct_change().fillna(0.0)
    total_return = clean.iloc[-1] / clean.iloc[0] - 1.0
    years = max(len(clean) / periods_per_year, 1 / periods_per_year)
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1.0

    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    max_drawdown = drawdown.min()

    vol = returns.std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else 0.0
    downside = returns[returns < 0].std(ddof=0) * math.sqrt(periods_per_year)
    sortino = (returns.mean() * periods_per_year) / downside if downside > 0 else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    trades = trades if trades is not None else pd.DataFrame()
    if not trades.empty and "return" in trades:
        trade_returns = trades["return"].astype(float)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = len(wins) / len(trade_returns) if len(trade_returns) else 0.0
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    monthly = clean.resample("ME").last().pct_change().dropna()
    stability = float((monthly > 0).mean()) if len(monthly) else 0.0

    return {
        "total_return": round(_safe_float(total_return), 10),
        "cagr": _safe_float(cagr),
        "max_drawdown": _safe_float(max_drawdown),
        "volatility": _safe_float(vol),
        "sharpe": _safe_float(sharpe),
        "sortino": _safe_float(sortino),
        "calmar": _safe_float(calmar),
        "win_rate": _safe_float(win_rate),
        "profit_factor": _safe_float(profit_factor, 10.0),
        "stability": _safe_float(stability),
    }


def _bounded(value: float, low: float, high: float) -> float:
    value = _safe_float(value)
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def score_strategy(kpis: dict[str, float]) -> float:
    """Composite 0-100 strategy score balancing return, risk, robustness and costs."""
    cagr_score = _bounded(kpis.get("cagr", 0.0), -0.05, 0.25)
    dd_score = 1.0 - _bounded(abs(kpis.get("max_drawdown", 0.0)), 0.05, 0.55)
    sharpe_score = _bounded(kpis.get("sharpe", 0.0), 0.0, 2.0)
    sortino_score = _bounded(kpis.get("sortino", 0.0), 0.0, 3.0)
    calmar_score = _bounded(kpis.get("calmar", 0.0), 0.0, 2.5)
    stability_score = _bounded(kpis.get("stability", 0.0), 0.35, 0.75)
    turnover_penalty = _bounded(kpis.get("trades_per_year", 0.0), 12, 160)

    score = (
        0.25 * cagr_score
        + 0.20 * dd_score
        + 0.12 * sharpe_score
        + 0.08 * sortino_score
        + 0.15 * calmar_score
        + 0.15 * stability_score
        + 0.05 * (1.0 - turnover_penalty)
    )
    return round(max(0.0, min(100.0, score * 100)), 2)


def drawdown_series(equity: pd.Series) -> pd.Series:
    clean = equity.dropna().astype(float)
    return clean / clean.cummax() - 1.0
