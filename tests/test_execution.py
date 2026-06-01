import pandas as pd

from trading_agents_dashboard.execution import (
    ReadinessRules,
    assess_readiness,
    build_position_plan,
    parse_holdings_text,
)


def test_assess_readiness_marks_ready_when_strategy_passes_rules():
    kpis = {"score": 82, "max_drawdown": -0.18, "trades_per_year": 8, "calmar": 1.2}
    assessment = assess_readiness(kpis, ReadinessRules(min_score=70, max_drawdown=0.25, max_trades_per_year=24, min_calmar=0.8))

    assert assessment.status == "READY"
    assert assessment.passed is True
    assert assessment.failed_checks == []


def test_assess_readiness_rejects_strategy_when_risk_rules_fail():
    kpis = {"score": 55, "max_drawdown": -0.42, "trades_per_year": 90, "calmar": 0.3}
    assessment = assess_readiness(kpis, ReadinessRules())

    assert assessment.status == "REJECT"
    assert assessment.passed is False
    assert len(assessment.failed_checks) >= 3


def test_parse_holdings_text_accepts_lines_and_commas():
    holdings = parse_holdings_text("SPY: 3\nQQQ, 2.5\nGLD 1")

    assert holdings == {"SPY": 3.0, "QQQ": 2.5, "GLD": 1.0}


def test_build_position_plan_calculates_target_trade_values_and_shares():
    prices = pd.Series({"SPY": 100.0, "GLD": 50.0, "SHY": 80.0})
    target_weights = pd.Series({"SPY": 0.6, "GLD": 0.4, "SHY": 0.0})
    current_holdings = {"SPY": 4.0, "GLD": 2.0}

    plan = build_position_plan(
        prices=prices,
        target_weights=target_weights,
        portfolio_value=1_000,
        current_holdings=current_holdings,
        min_trade_value=25,
    )

    spy = plan[plan["asset"] == "SPY"].iloc[0]
    gld = plan[plan["asset"] == "GLD"].iloc[0]

    assert spy["target_value"] == 600.0
    assert spy["current_value"] == 400.0
    assert spy["trade_value"] == 200.0
    assert spy["trade_shares"] == 2.0
    assert gld["trade_value"] == 300.0
    assert "BUY" in set(plan["action"])
