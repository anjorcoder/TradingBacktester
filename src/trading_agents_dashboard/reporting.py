from __future__ import annotations

import pandas as pd


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def compact_scoreboard(board: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "agent",
        "score",
        "cagr",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "win_rate",
        "profit_factor",
        "trades_per_year",
        "current_signal",
    ]
    return board[[col for col in columns if col in board.columns]].copy()
