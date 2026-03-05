"""
RITA Core — Goal Engine
Step 1: Set financial goal anchored to real Nifty 50 historical data.
Step 8: Update goal based on backtest results (closed feedback loop).
"""

from .data_loader import RISK_FREE_RATE


def set_goal(
    target_return_pct: float,
    time_horizon_days: int,
    risk_tolerance: str,
    historical_stats: dict,
) -> dict:
    """
    Validate and set a financial goal against historical Nifty 50 performance.

    Returns a goal dict with feasibility assessment and derived metrics.
    """
    hist_cagr = historical_stats["cagr_pct"]
    hist_best = historical_stats["best_year_pct"]
    hist_worst = historical_stats["worst_year_pct"]
    hist_sharpe = historical_stats["sharpe_ratio"]

    # Annualize the target
    years = time_horizon_days / 365.0
    annualized_target = ((1 + target_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Feasibility check
    if annualized_target <= hist_cagr * 0.5:
        feasibility = "conservative"
        feasibility_note = f"Target is well below the {hist_cagr:.1f}% historical CAGR — very achievable."
    elif annualized_target <= hist_cagr * 1.2:
        feasibility = "realistic"
        feasibility_note = f"Target is in line with the {hist_cagr:.1f}% historical CAGR — realistic with active strategy."
    elif annualized_target <= hist_best:
        feasibility = "ambitious"
        feasibility_note = f"Target is above historical CAGR ({hist_cagr:.1f}%) but within best-year range ({hist_best:.1f}%)."
    else:
        feasibility = "unrealistic"
        feasibility_note = (
            f"Target exceeds historical best year ({hist_best:.1f}%). "
            f"Consider reducing to {hist_cagr * 1.3:.1f}% annualized."
        )

    # Required monthly return
    monthly_return = ((1 + target_return_pct / 100) ** (30 / time_horizon_days) - 1) * 100

    # Risk-adjusted benchmark
    risk_map = {
        "conservative": hist_cagr * 0.7,
        "moderate": hist_cagr,
        "aggressive": hist_cagr * 1.3,
    }
    suggested_target = risk_map.get(risk_tolerance, hist_cagr)

    return {
        "target_return_pct": target_return_pct,
        "time_horizon_days": time_horizon_days,
        "years": round(years, 2),
        "risk_tolerance": risk_tolerance,
        "annualized_target_pct": round(annualized_target, 2),
        "required_monthly_return_pct": round(monthly_return, 3),
        "feasibility": feasibility,
        "feasibility_note": feasibility_note,
        "historical_cagr_pct": hist_cagr,
        "historical_best_year_pct": hist_best,
        "historical_worst_year_pct": hist_worst,
        "historical_sharpe": hist_sharpe,
        "suggested_realistic_target_pct": round(suggested_target, 1),
        "strategy_constraints": {
            "min_sharpe_ratio": 1.0,
            "max_drawdown_pct": -10.0,
        },
    }


def update_goal_from_results(original_goal: dict, backtest_results: dict) -> dict:
    """
    Step 8: Compare backtest performance to original goal and produce a revised goal.
    Closes the reinforcement learning feedback loop.
    """
    actual_return = backtest_results["performance"].get("portfolio_total_return_pct", 0)
    actual_sharpe = backtest_results["performance"].get("sharpe_ratio", 0)
    actual_mdd = backtest_results["performance"].get("max_drawdown_pct", 0)
    constraints_met = backtest_results["performance"].get("constraints_met", False)

    target = original_goal.get("target_return_pct", 0)
    delta = actual_return - target

    if delta >= 5:
        assessment = "exceeded"
        new_target = round(target * 1.1, 1)
        recommendation = f"Strategy exceeded goal by {delta:.1f}%. Raise target to {new_target}% for next cycle."
    elif delta >= -5:
        assessment = "met"
        new_target = round(target, 1)
        recommendation = "Strategy met the goal within tolerance. Maintain current target for next cycle."
    elif delta >= -15:
        assessment = "missed"
        new_target = round(actual_return * 0.9, 1)
        recommendation = (
            f"Strategy missed goal by {abs(delta):.1f}%. "
            f"Revise target to {new_target}% based on observed performance."
        )
    else:
        assessment = "significantly_missed"
        new_target = round(actual_return * 0.8, 1)
        recommendation = (
            f"Significant underperformance ({delta:.1f}%). "
            f"Reset target to {new_target}% — consider retraining model with different parameters."
        )

    constraint_notes = []
    if actual_sharpe <= 1.0:
        constraint_notes.append(f"Sharpe ratio ({actual_sharpe:.2f}) did not meet target > 1.0")
    if actual_mdd < -10:
        constraint_notes.append(f"Max drawdown ({actual_mdd:.1f}%) exceeded -10% threshold")

    return {
        "original_goal": original_goal,
        "backtest_summary": {
            "actual_return_pct": actual_return,
            "actual_sharpe": actual_sharpe,
            "actual_max_drawdown_pct": actual_mdd,
            "constraints_met": constraints_met,
        },
        "goal_delta_pct": round(delta, 2),
        "assessment": assessment,
        "revised_target_pct": new_target,
        "recommendation": recommendation,
        "constraint_violations": constraint_notes,
    }
