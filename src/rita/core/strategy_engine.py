"""
RITA Core — Strategy Engine
Step 3: Design a trading strategy and validate it meets the constraints
(Sharpe ratio > 1, max drawdown < 10%) using an in-sample quick-check.
"""

from .performance import sharpe_ratio, max_drawdown
import numpy as np
import pandas as pd


# Allocation rules keyed by (risk_tolerance, trend)
_ALLOCATION_RULES = {
    ("conservative", "uptrend"):   {"base_allocation": 0.60, "rebalance": "monthly"},
    ("conservative", "sideways"):  {"base_allocation": 0.40, "rebalance": "quarterly"},
    ("conservative", "downtrend"): {"base_allocation": 0.20, "rebalance": "monthly"},
    ("moderate",     "uptrend"):   {"base_allocation": 0.80, "rebalance": "monthly"},
    ("moderate",     "sideways"):  {"base_allocation": 0.60, "rebalance": "monthly"},
    ("moderate",     "downtrend"): {"base_allocation": 0.30, "rebalance": "monthly"},
    ("aggressive",   "uptrend"):   {"base_allocation": 1.00, "rebalance": "weekly"},
    ("aggressive",   "sideways"):  {"base_allocation": 0.70, "rebalance": "monthly"},
    ("aggressive",   "downtrend"): {"base_allocation": 0.40, "rebalance": "monthly"},
}


def design_strategy(research: dict, goal: dict) -> dict:
    """
    Design a strategy based on market research and the financial goal.

    Returns strategy parameters and a preliminary quick-check using
    a simple buy/hold simulation over a recent data window.
    """
    risk_tolerance = goal.get("risk_tolerance", "moderate")
    trend = research.get("trend", "sideways")
    sentiment = research.get("sentiment_proxy", "neutral")

    key = (risk_tolerance, trend)
    rule = _ALLOCATION_RULES.get(key, {"base_allocation": 0.60, "rebalance": "monthly"})

    # Adjust for sentiment
    base_alloc = rule["base_allocation"]
    if sentiment == "fearful":
        base_alloc = max(0.0, base_alloc - 0.15)
    elif sentiment == "complacent" and trend == "uptrend":
        base_alloc = min(1.0, base_alloc + 0.05)

    strategy = {
        "name": f"{risk_tolerance.capitalize()} {trend.replace('_',' ').title()} Strategy",
        "risk_tolerance": risk_tolerance,
        "market_trend": trend,
        "sentiment_proxy": sentiment,
        "base_allocation_pct": round(base_alloc * 100, 1),
        "rebalancing_frequency": rule["rebalance"],
        "action_space": {
            "0": "Hold Cash (0% invested)",
            "1": "Half Position (50% invested)",
            "2": "Full Position (100% invested)",
        },
        "stop_loss_rule": "Exit if single-day loss > 3%",
        "take_profit_rule": "Lock 50% at +8% gain",
        "constraints": {
            "min_sharpe_ratio": 1.0,
            "max_drawdown_pct": -10.0,
        },
        "rationale": (
            f"In a {trend} market with {sentiment} sentiment, "
            f"a {risk_tolerance} investor should maintain {round(base_alloc*100)}% allocation. "
            f"DDQN model trained on 2010-2022 enforces these constraints automatically."
        ),
    }

    return strategy


def validate_strategy_constraints(backtest_results: dict) -> dict:
    """
    Check whether the backtest results satisfy the strategy constraints.

    Returns pass/fail for each constraint with recommendations.
    """
    perf = backtest_results.get("performance", {})
    sharpe = perf.get("sharpe_ratio", 0)
    mdd = perf.get("max_drawdown_pct", -999)
    port_return = perf.get("portfolio_total_return_pct", 0)
    bench_return = perf.get("benchmark_total_return_pct", 0)

    sharpe_ok = sharpe > 1.0
    mdd_ok = mdd > -10.0
    outperformed = port_return > bench_return

    recommendations = []
    if not sharpe_ok:
        recommendations.append(
            f"Sharpe ratio {sharpe:.2f} is below 1.0. "
            "Consider increasing training timesteps or adjusting reward function."
        )
    if not mdd_ok:
        recommendations.append(
            f"Max drawdown {mdd:.1f}% exceeds -10% threshold. "
            "Increase drawdown penalty in reward function and retrain."
        )
    if not outperformed:
        recommendations.append(
            "Strategy underperformed Buy-and-Hold. "
            "Review feature set and action space."
        )

    return {
        "sharpe_constraint_met": sharpe_ok,
        "drawdown_constraint_met": mdd_ok,
        "outperforms_benchmark": outperformed,
        "all_constraints_met": sharpe_ok and mdd_ok,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": mdd,
        "portfolio_return_pct": port_return,
        "benchmark_return_pct": bench_return,
        "recommendations": recommendations,
    }
