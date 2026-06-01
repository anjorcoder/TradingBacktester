from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "EEM", "GLD", "TLT", "SHY"]


def load_prices(tickers: list[str] | None = None, period: str = "10y") -> pd.DataFrame:
    """Load adjusted close prices from yfinance; fall back to deterministic demo data."""
    tickers = tickers or DEFAULT_TICKERS
    try:
        import yfinance as yf

        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
            timeout=15,
        )
        if data.empty:
            return demo_prices(tickers)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", axis=1, level=1)
        else:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna(how="all").ffill().dropna(axis=1, how="any")
        if len(prices) > 100 and len(prices.columns) > 0:
            return prices
    except Exception:
        pass
    return demo_prices(tickers)


def demo_prices(tickers: list[str] | None = None, rows: int = 2520) -> pd.DataFrame:
    """Deterministic synthetic market data for offline demos/tests."""
    tickers = tickers or DEFAULT_TICKERS
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="B")
    data = {}
    for idx, ticker in enumerate(tickers):
        drift = 0.00015 + idx * 0.00002
        vol = 0.007 + idx * 0.001
        cycle = 0.0008 * np.sin(np.linspace(0, 12 + idx, rows))
        returns = drift + cycle + rng.normal(0, vol, rows)
        data[ticker] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)
