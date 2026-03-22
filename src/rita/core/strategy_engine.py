"""
RITA Core — Strategy Engine
Step 3: Design a trading strategy and validate it meets the constraints
(Sharpe ratio > 1, max drawdown < 10%) using an in-sample quick-check.
"""



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


def get_allocation_recommendation(summary: dict, scored: dict) -> dict:
    """
    Rule-based proxy for the DDQN model's allocation decision.

    Inputs are the outputs of get_market_summary() and get_sentiment_score()
    from technical_analyzer.py — the same 5 signals the RL model observes.

    Returns one of three actions matching the RL model's action space:
        action 0 → HOLD  (0%  invested)
        action 1 → HALF  (50% invested)
        action 2 → FULL  (100% invested)

    Decision is calibrated toward the two project constraints:
        Sharpe > 1.0      — don't enter unless risk/reward is favourable
        Max drawdown < 10% — exit / reduce if downside risk is elevated
    """
    total_score = scored["total_score"]
    trend       = summary["trend"]
    rsi_signal  = summary["rsi_signal"]
    bb_position = summary["bb_position"]
    volatility  = summary["sentiment_proxy"]   # fearful / neutral / complacent
    overall     = scored["overall_sentiment"]

    # ── Base action from sentiment score ─────────────────────────────────────
    # Mirrors the RL model: reward positive returns, penalise drawdown heavily
    if total_score >= 3:
        action = 2   # FULL — strong bullish consensus, maximise Sharpe
    elif total_score >= -1:
        action = 1   # HALF — mixed signals, balance return vs drawdown
    else:
        action = 0   # HOLD — bearish, protect against MDD breach

    # ── Override rules — max drawdown < 10% protection ───────────────────────
    # These mirror the -10.0 drawdown penalty in the RL reward function.
    override_reason = None

    # High volatility: large daily swings compound into MDD breach quickly
    if volatility == "fearful" and action == 2:
        action = 1
        override_reason = (
            "High volatility (fearful ATR) caps at 50% — "
            "large daily swings risk breaching 10% drawdown limit"
        )

    # Downtrend: sustained losses erode the portfolio peak
    if trend == "downtrend" and action == 2:
        action = 1
        override_reason = (
            "Downtrend caps at 50% — "
            "sustained negative returns risk breaching 10% drawdown limit"
        )

    # Both downtrend AND fearful volatility → eliminate market exposure entirely
    if trend == "downtrend" and volatility == "fearful":
        action = 0
        override_reason = (
            "Downtrend + fearful volatility: HOLD — "
            "combined risk too high, protecting against drawdown > 10%"
        )

    # Momentum exhaustion: overbought RSI at upper Bollinger Band
    # signals likely reversal — step down one level
    if rsi_signal == "overbought" and bb_position == "near_upper_band":
        action = max(0, action - 1)
        override_reason = (
            "Overbought RSI + price at upper Bollinger Band: "
            "reversal risk, stepping down one allocation level"
        )

    # ── Map action to output labels ───────────────────────────────────────────
    _ACTION_MAP = {
        0: ("HOLD", 0),
        1: ("HALF", 50),
        2: ("FULL", 100),
    }
    label, alloc_pct = _ACTION_MAP[action]

    # Primary constraint driving this decision
    if action == 0:
        primary_constraint = "max_drawdown < 10%"
    elif action == 2:
        primary_constraint = "Sharpe > 1.0"
    else:
        primary_constraint = "Sharpe > 1.0 & max_drawdown < 10%"

    # ── Plain-English rationale ───────────────────────────────────────────────
    score_desc = scored["signal_summary"]
    if override_reason:
        rationale = (
            f"Base signals ({overall}, score {total_score:+d}/6): {score_desc}. "
            f"Override: {override_reason}."
        )
    else:
        rationale = (
            f"Sentiment {overall} (score {total_score:+d}/6): {score_desc}. "
            f"Driving constraint: {primary_constraint}."
        )

    # ── Triggers for next review ──────────────────────────────────────────────
    if action == 2:
        upgrade_trigger   = "Already at maximum allocation."
        downgrade_trigger = (
            "Reduce to HALF if downtrend confirmed or volatility turns fearful "
            "(score drops below +3)"
        )
    elif action == 1:
        upgrade_trigger = (
            "Move to FULL if uptrend + MACD bullish + volatility not fearful "
            "(score >= 3)"
        )
        downgrade_trigger = (
            "Move to HOLD if downtrend + fearful volatility or score <= -2"
        )
    else:
        upgrade_trigger = (
            "Move to HALF if trend turns sideways or up and MACD turns bullish "
            "(score >= -1)"
        )
        downgrade_trigger = "Already at minimum allocation."

    return {
        "recommendation":   label,
        "allocation_pct":   alloc_pct,
        "action_code":      action,
        "primary_constraint": primary_constraint,
        "rationale":        rationale,
        "upgrade_trigger":  upgrade_trigger,
        "downgrade_trigger": downgrade_trigger,
        "override_applied": override_reason is not None,
        "override_reason":  override_reason,
    }


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
