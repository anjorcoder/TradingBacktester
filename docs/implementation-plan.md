# Trading Agents Dashboard Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Bouw een Streamlit MVP waarin meerdere trading strategy agents worden gebacktest en gerankt op KPI's.

**Architecture:** Python package met losse modules voor data, strategieën, backtesting, KPI/scoring en dashboard UI. Strategieën produceren weight frames; de backtest-engine voert signalen één bar later uit om look-ahead bias te beperken.

**Tech Stack:** Python, pandas, numpy, Streamlit, Plotly, yfinance, pytest.

---

## Taken

1. Projectstructuur en pyproject aanmaken.
2. Tests schrijven voor KPI's, scoring, backtest-output en strategie-weights.
3. KPI/scoring-module implementeren.
4. Strategie-agents implementeren.
5. Backtest-engine implementeren met kosten/slippage en trade-log.
6. Data-loader implementeren met yfinance en offline demo fallback.
7. Streamlit app bouwen met scorebord, equity curves, drawdowns en agentdetails.
8. Tests draaien en README documenteren.

## Safeguards

- Geen live orderuitvoering.
- Signalen worden geshift om look-ahead bias te beperken.
- Kosten + slippage worden standaard meegenomen.
- Dashboard vermeldt expliciet dat het research is, geen financieel advies.
