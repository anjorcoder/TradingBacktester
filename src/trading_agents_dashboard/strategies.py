from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class StrategyAgent:
    name = "Base Strategy Agent"
    description = "Base class"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def current_signal(self, prices: pd.DataFrame) -> str:
        weights = self.generate_weights(prices).iloc[-1]
        active = weights[weights > 0]
        if active.empty:
            return "Cash / geen positie"
        return ", ".join(f"{asset}: {weight:.0%}" for asset, weight in active.sort_values(ascending=False).items())


def _empty_weights(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


def _normalize_rows(weights: pd.DataFrame) -> pd.DataFrame:
    sums = weights.sum(axis=1).replace(0, np.nan)
    return weights.div(sums, axis=0).fillna(0.0).clip(lower=0.0, upper=1.0)


@dataclass
class SMATrendAgent(StrategyAgent):
    window: int = 200
    name: str = "SMA Trend Agent"
    description: str = "Equal-weight assets die boven hun simple moving average handelen."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sma = prices.rolling(self.window, min_periods=max(5, self.window // 3)).mean()
        signal = prices > sma
        return _normalize_rows(signal.astype(float))


@dataclass
class DualMomentumAgent(StrategyAgent):
    lookback: int = 126
    top_n: int = 2
    name: str = "Dual Momentum Agent"
    description: str = "Roteert naar assets met sterkste positieve momentum."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(self.lookback)
        weights = _empty_weights(prices)
        for date, row in momentum.iterrows():
            winners = row[row > 0].sort_values(ascending=False).head(self.top_n)
            if not winners.empty:
                weights.loc[date, winners.index] = 1.0 / len(winners)
        return weights


@dataclass
class RSIMeanReversionAgent(StrategyAgent):
    rsi_window: int = 14
    trend_window: int = 100
    name: str = "RSI Mean Reversion Agent"
    description: str = "Koopt oversold dips binnen een stijgende trend."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_window, min_periods=self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_window, min_periods=self.rsi_window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        trend = prices > prices.rolling(self.trend_window, min_periods=max(10, self.trend_window // 3)).mean()
        signal = (rsi < 35) & trend
        return _normalize_rows(signal.astype(float))


@dataclass
class BreakoutAgent(StrategyAgent):
    lookback: int = 55
    name: str = "Breakout Agent"
    description: str = "Koopt nieuwe highs over de gekozen lookbackperiode."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        prior_high = prices.shift(1).rolling(self.lookback, min_periods=max(10, self.lookback // 3)).max()
        signal = prices >= prior_high
        return _normalize_rows(signal.astype(float))


@dataclass
class LowVolatilityAgent(StrategyAgent):
    vol_window: int = 63
    top_n: int = 2
    name: str = "Low Volatility Agent"
    description: str = "Selecteert assets met laagste gerealiseerde volatiliteit."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vol = prices.pct_change().rolling(self.vol_window, min_periods=max(10, self.vol_window // 3)).std()
        trend = prices > prices.rolling(100, min_periods=30).mean()
        weights = _empty_weights(prices)
        for date, row in vol.iterrows():
            eligible = row[trend.loc[date]].dropna().sort_values().head(self.top_n)
            if not eligible.empty:
                weights.loc[date, eligible.index] = 1.0 / len(eligible)
        return weights


@dataclass
class DefensiveRotationAgent(StrategyAgent):
    momentum_lookback: int = 126
    defensive_assets: tuple[str, ...] = ("GLD", "TLT", "SHY")
    name: str = "Defensive Rotation Agent"
    description: str = "Risk-on naar beste momentum; risk-off naar defensieve assets indien beschikbaar."

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(self.momentum_lookback)
        weights = _empty_weights(prices)
        defensive = [asset for asset in self.defensive_assets if asset in prices.columns]
        for date, row in momentum.iterrows():
            positive = row[row > 0].sort_values(ascending=False)
            if not positive.empty:
                weights.loc[date, positive.index[0]] = 1.0
            elif defensive:
                available = row[defensive].dropna().sort_values(ascending=False)
                if not available.empty:
                    weights.loc[date, available.index[0]] = 1.0
        return weights


def default_agents() -> list[StrategyAgent]:
    return [
        SMATrendAgent(),
        DualMomentumAgent(),
        RSIMeanReversionAgent(),
        BreakoutAgent(),
        LowVolatilityAgent(),
        DefensiveRotationAgent(),
    ]
