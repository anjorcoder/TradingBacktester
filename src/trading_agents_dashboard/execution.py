from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class ReadinessRules:
    min_score: float = 65.0
    max_drawdown: float = 0.30
    max_trades_per_year: float = 36.0
    min_calmar: float = 0.5


@dataclass(frozen=True)
class ReadinessAssessment:
    status: str
    passed: bool
    failed_checks: list[str]
    warnings: list[str]


def assess_readiness(kpis: dict[str, float], rules: ReadinessRules | None = None) -> ReadinessAssessment:
    """Assess whether a strategy is suitable for paper/live operational follow-up."""
    rules = rules or ReadinessRules()
    failed: list[str] = []
    warnings: list[str] = []

    score = float(kpis.get("score", 0.0))
    max_drawdown = abs(float(kpis.get("max_drawdown", 0.0)))
    trades_per_year = float(kpis.get("trades_per_year", 0.0))
    calmar = float(kpis.get("calmar", 0.0))

    if score < rules.min_score:
        failed.append(f"Score {score:.1f} is lager dan minimum {rules.min_score:.1f}.")
    if max_drawdown > rules.max_drawdown:
        failed.append(f"Max drawdown {max_drawdown:.1%} is hoger dan limiet {rules.max_drawdown:.1%}.")
    if trades_per_year > rules.max_trades_per_year:
        failed.append(f"Trades/jaar {trades_per_year:.1f} is hoger dan limiet {rules.max_trades_per_year:.1f}.")
    if calmar < rules.min_calmar:
        failed.append(f"Calmar {calmar:.2f} is lager dan minimum {rules.min_calmar:.2f}.")

    if trades_per_year > 12:
        warnings.append("Controleer of rebalance-frequentie, spread en brokerkosten realistisch zijn.")
    if max_drawdown > 0.20:
        warnings.append("Drawdown is substantieel; bepaal vooraf maximale positieomvang en stopregels.")

    if not failed:
        status = "READY"
    elif len(failed) <= 2 and score >= rules.min_score * 0.8:
        status = "PAPERTRADE"
    else:
        status = "REJECT"

    return ReadinessAssessment(status=status, passed=not failed, failed_checks=failed, warnings=warnings)


def parse_holdings_text(text: str) -> dict[str, float]:
    """Parse simple holdings input like 'SPY: 3\nQQQ, 2\nGLD 1'."""
    holdings: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[:,;\s]+", line) if part]
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        try:
            shares = float(parts[1])
        except ValueError:
            continue
        holdings[ticker] = shares
    return holdings


def build_position_plan(
    prices: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    current_holdings: dict[str, float] | None = None,
    min_trade_value: float = 25.0,
) -> pd.DataFrame:
    """Build a practical rebalance plan from target weights and current holdings.

    This is not order execution. It calculates approximate target values and share
    differences using latest close prices.
    """
    current_holdings = current_holdings or {}
    prices = prices.dropna().astype(float)
    weights = target_weights.reindex(prices.index).fillna(0.0).clip(lower=0.0)
    if weights.sum() > 1.0:
        weights = weights / weights.sum()

    rows = []
    for asset in prices.index:
        price = float(prices[asset])
        target_weight = float(weights.get(asset, 0.0))
        target_value = round(float(portfolio_value) * target_weight, 2)
        current_shares = float(current_holdings.get(asset, 0.0))
        current_value = round(current_shares * price, 2)
        trade_value = round(target_value - current_value, 2)
        trade_shares = round(trade_value / price, 6) if price > 0 else 0.0
        if abs(trade_value) < min_trade_value:
            action = "HOLD"
            trade_value = 0.0
            trade_shares = 0.0
        elif trade_value > 0:
            action = "BUY"
        else:
            action = "SELL"
        rows.append(
            {
                "asset": asset,
                "price": round(price, 4),
                "target_weight": target_weight,
                "target_value": target_value,
                "current_shares": current_shares,
                "current_value": current_value,
                "trade_value": trade_value,
                "trade_shares": trade_shares,
                "action": action,
            }
        )

    return pd.DataFrame(rows).sort_values(["action", "asset"]).reset_index(drop=True)
