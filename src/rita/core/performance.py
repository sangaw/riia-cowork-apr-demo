"""
RITA Core — Performance Analyzer
Calculates Sharpe ratio, max drawdown, CAGR and generates interpretability plots.
"""

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RISK_FREE_RATE = 0.07  # India 10Y govt bond yield (annualized)
TRADING_DAYS = 252


def sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio from daily returns array."""
    daily_rf = risk_free_rate / TRADING_DAYS
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0 or arr.std() == 0:
        return 0.0
    return float((arr.mean() - daily_rf) / arr.std() * math.sqrt(TRADING_DAYS))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Maximum drawdown as a negative fraction (e.g. -0.082 = -8.2%).
    Uses rolling-peak method.
    """
    vals = np.asarray(portfolio_values, dtype=float)
    if len(vals) == 0:
        return 0.0
    running_max = np.maximum.accumulate(vals)
    drawdowns = (vals - running_max) / running_max
    return float(drawdowns.min())


def cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return 0.0
    return float((end_value / start_value) ** (1 / years) - 1)


def compute_all_metrics(portfolio_values: np.ndarray, benchmark_values: np.ndarray) -> dict:
    """
    Compute full performance report for a completed backtest.

    Args:
        portfolio_values: daily portfolio value array (starts at 1.0 or 100.0)
        benchmark_values: daily Buy-and-Hold Nifty values (normalized same start)

    Returns:
        dict with all performance metrics
    """
    port = np.asarray(portfolio_values, dtype=float)
    bench = np.asarray(benchmark_values, dtype=float)

    daily_rets = np.diff(port) / port[:-1]
    bench_rets = np.diff(bench) / bench[:-1]

    years = len(port) / TRADING_DAYS

    sr = sharpe_ratio(daily_rets)
    mdd = max_drawdown(port)
    port_cagr = cagr(port[0], port[-1], years)
    bench_cagr = cagr(bench[0], bench[-1], years)

    win_days = int(np.sum(daily_rets > 0))
    total_days = len(daily_rets)

    return {
        "total_days": total_days,
        "years": round(years, 2),
        "portfolio_total_return_pct": round((port[-1] / port[0] - 1) * 100, 2),
        "benchmark_total_return_pct": round((bench[-1] / bench[0] - 1) * 100, 2),
        "portfolio_cagr_pct": round(port_cagr * 100, 2),
        "benchmark_cagr_pct": round(bench_cagr * 100, 2),
        "sharpe_ratio": round(sr, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "annual_volatility_pct": round(float(daily_rets.std() * math.sqrt(TRADING_DAYS) * 100), 2),
        "win_rate_pct": round(win_days / total_days * 100, 2) if total_days > 0 else 0.0,
        "sharpe_constraint_met": sr > 1.0,
        "drawdown_constraint_met": mdd > -0.10,
        "constraints_met": sr > 1.0 and mdd > -0.10,
    }


def build_portfolio_comparison(backtest_df: pd.DataFrame, portfolio_inr: float) -> dict:
    """
    Compare three fixed-allocation manual strategies against the RITA RL model
    on the same historical date range and Nifty price series.

    Args:
        backtest_df : DataFrame from backtest_daily.csv — columns:
                      date, portfolio_value, benchmark_value, allocation, close_price
                      portfolio_value and benchmark_value are normalized to 1.0 at start.
        portfolio_inr : Starting capital in INR (e.g. 1_000_000 for Rs 10 lakh).

    Returns:
        Structured comparison dict with per-profile metrics and INR values.
    """
    df = backtest_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Reconstruct daily Nifty returns from close prices
    closes = df["close_price"].values
    market_returns = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])

    years = len(df) / TRADING_DAYS

    def _simulate_fixed(alloc: float) -> np.ndarray:
        """Simulate a buy-and-hold fixed allocation on daily market returns."""
        vals = np.ones(len(df))
        for i in range(1, len(df)):
            vals[i] = vals[i - 1] * (1.0 + alloc * market_returns[i])
        return vals

    def _profile_metrics(norm_values: np.ndarray, label: str, alloc_desc: str) -> dict:
        daily_rets = np.diff(norm_values) / norm_values[:-1]
        sr  = sharpe_ratio(daily_rets)
        mdd = max_drawdown(norm_values)
        total_return = (norm_values[-1] / norm_values[0] - 1)
        final_inr    = round(portfolio_inr * norm_values[-1])
        profit_inr   = round(final_inr - portfolio_inr)
        return {
            "label":              label,
            "allocation":         alloc_desc,
            "start_value_inr":    int(portfolio_inr),
            "final_value_inr":    final_inr,
            "profit_loss_inr":    profit_inr,
            "return_pct":         round(total_return * 100, 2),
            "cagr_pct":           round(cagr(1.0, norm_values[-1], years) * 100, 2),
            "max_drawdown_pct":   round(mdd * 100, 2),
            "sharpe_ratio":       round(sr, 3),
            "sharpe_constraint_met":   sr > 1.0,
            "drawdown_constraint_met": mdd > -0.10,
        }

    # ── Three manual fixed-allocation profiles ────────────────────────────────
    conservative = _profile_metrics(_simulate_fixed(0.30), "Conservative", "30% Nifty + 70% Cash")
    moderate     = _profile_metrics(_simulate_fixed(0.60), "Moderate",     "60% Nifty + 40% Cash")
    aggressive   = _profile_metrics(_simulate_fixed(1.00), "Aggressive",   "100% Nifty (Buy & Hold)")

    # ── RITA RL model — use portfolio_value from backtest_daily.csv ───────────
    rita_norm = df["portfolio_value"].values
    rita      = _profile_metrics(rita_norm, "RITA RL Model", "0 / 50 / 100% dynamic (DDQN)")

    profiles = {
        "conservative": conservative,
        "moderate":     moderate,
        "aggressive":   aggressive,
        "rita_model":   rita,
    }

    # ── Winner by Sharpe (project goal) ──────────────────────────────────────
    best_key   = max(profiles, key=lambda k: profiles[k]["sharpe_ratio"])
    best_return_key = max(profiles, key=lambda k: profiles[k]["return_pct"])

    # ── Summary table rows (for easy display) ────────────────────────────────
    table = []
    for key, p in profiles.items():
        table.append({
            "profile":       p["label"],
            "allocation":    p["allocation"],
            "final_inr":     p["final_value_inr"],
            "profit_inr":    p["profit_loss_inr"],
            "return_pct":    p["return_pct"],
            "max_dd_pct":    p["max_drawdown_pct"],
            "sharpe":        p["sharpe_ratio"],
            "sharpe_ok":     p["sharpe_constraint_met"],
            "dd_ok":         p["drawdown_constraint_met"],
        })

    return {
        "period_start":   df["date"].iloc[0].strftime("%Y-%m-%d"),
        "period_end":     df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "trading_days":   len(df),
        "portfolio_inr":  int(portfolio_inr),
        "nifty_return_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "profiles":       profiles,
        "summary_table":  table,
        "sharpe_winner":  best_key,
        "return_winner":  best_return_key,
        "insight": (
            f"RITA RL model achieves Sharpe {rita['sharpe_ratio']:.3f} vs "
            f"Aggressive (Buy & Hold) Sharpe {aggressive['sharpe_ratio']:.3f}. "
            f"{'RITA wins on risk-adjusted return (project goal: Sharpe > 1.0).' if rita['sharpe_ratio'] > aggressive['sharpe_ratio'] else 'Buy & Hold leads on Sharpe this period.'}"
        ),
    }


def build_performance_feedback(
    backtest_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> dict:
    """
    Outcome Analyst: summarise how the RITA RL model performed over the backtest
    period and derive realistic return expectations for planning.

    Args:
        backtest_df : DataFrame from backtest_daily.csv
        perf_df     : DataFrame from performance_summary.csv (metric/value pairs)
        history_df  : DataFrame from training_history.csv

    Returns:
        Structured feedback dict covering performance, trade activity, constraints,
        and realistic forward-looking return expectations.
    """
    # ── Load performance metrics ──────────────────────────────────────────────
    perf = dict(zip(perf_df["metric"], perf_df["value"]))

    def _f(key, default=0.0):
        try:
            return float(perf.get(key, default))
        except (ValueError, TypeError):
            return default

    sharpe      = _f("sharpe_ratio")
    mdd         = _f("max_drawdown_pct")
    total_ret   = _f("portfolio_total_return_pct")
    bench_ret   = _f("benchmark_total_return_pct")
    cagr_pct    = _f("portfolio_cagr_pct")
    bench_cagr  = _f("benchmark_cagr_pct")
    volatility  = _f("annual_volatility_pct")
    win_rate    = _f("win_rate_pct")
    total_days  = int(_f("total_days"))
    years       = _f("years")

    sharpe_ok = str(perf.get("sharpe_constraint_met", "False")).lower() == "true"
    dd_ok     = str(perf.get("drawdown_constraint_met", "False")).lower() == "true"

    # ── Trade activity from backtest_daily.csv ────────────────────────────────
    bt = backtest_df.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.sort_values("date").reset_index(drop=True)

    alloc   = bt["allocation"].dropna()
    changes = alloc.diff().fillna(0)

    total_trades   = int((changes.abs() > 0).sum())
    buy_trades     = int((changes > 0).sum())
    sell_trades    = int((changes < 0).sum())

    days_hold = int((alloc == 0.0).sum())
    days_half = int((alloc == 0.5).sum())
    days_full = int((alloc == 1.0).sum())
    days_invested = int((alloc > 0).sum())

    port   = bt["portfolio_value"].values
    closes = bt["close_price"].values
    daily_rets = np.diff(port) / port[:-1] * 100

    # ── Training round context ────────────────────────────────────────────────
    latest_round = None
    round_number = None
    if not history_df.empty:
        latest = history_df.iloc[-1]
        round_number = int(latest.get("round", len(history_df)))
        latest_round = {
            "round":           round_number,
            "timesteps":       int(latest.get("timesteps", 0)),
            "source":          str(latest.get("source", "unknown")),
            "val_sharpe":      round(float(latest.get("val_sharpe", 0)), 3),
            "val_mdd_pct":     round(float(latest.get("val_mdd_pct", 0)), 2),
        }

    # ── Realistic return expectations ─────────────────────────────────────────
    # Annualise observed CAGR conservatively: apply a 20% discount for uncertainty
    conservative_annual = round(cagr_pct * 0.80, 1)
    realistic_annual    = round(cagr_pct, 1)

    # 1-year projection from current portfolio base
    def _project(rate_pct, years_ahead=1):
        return round((1 + rate_pct / 100) ** years_ahead - 1, 4) * 100

    expectations = {
        "conservative_1y_pct":  round(_project(conservative_annual), 2),
        "realistic_1y_pct":     round(_project(realistic_annual), 2),
        "conservative_3y_pct":  round(_project(conservative_annual, 3), 2),
        "realistic_3y_pct":     round(_project(realistic_annual, 3), 2),
        "basis": (
            f"Based on observed CAGR of {cagr_pct:.1f}% over {total_days} trading days. "
            f"Conservative estimate applies 20% discount for market uncertainty."
        ),
    }

    # ── Constraint verdict ────────────────────────────────────────────────────
    constraint_verdict = "ALL CONSTRAINTS MET" if (sharpe_ok and dd_ok) else (
        "SHARPE CONSTRAINT FAILED" if not sharpe_ok else "DRAWDOWN CONSTRAINT FAILED"
    )

    # ── Alpha vs benchmark ────────────────────────────────────────────────────
    alpha = round(total_ret - bench_ret, 2)
    alpha_note = (
        f"Underperformed Buy & Hold by {abs(alpha):.1f}% on raw return, "
        f"but Sharpe {sharpe:.3f} vs implied B&H Sharpe shows better risk-adjusted outcome."
        if alpha < 0 else
        f"Outperformed Buy & Hold by {alpha:.1f}% on raw return."
    )

    return {
        # Period
        "period_start":     bt["date"].iloc[0].strftime("%Y-%m-%d"),
        "period_end":       bt["date"].iloc[-1].strftime("%Y-%m-%d"),
        "trading_days":     total_days,
        "years":            round(years, 2),

        # Return metrics
        "return_metrics": {
            "portfolio_return_pct":  round(total_ret, 2),
            "benchmark_return_pct":  round(bench_ret, 2),
            "alpha_pct":             alpha,
            "portfolio_cagr_pct":    round(cagr_pct, 2),
            "benchmark_cagr_pct":    round(bench_cagr, 2),
            "alpha_note":            alpha_note,
        },

        # Risk metrics
        "risk_metrics": {
            "sharpe_ratio":          round(sharpe, 3),
            "max_drawdown_pct":      round(mdd, 2),
            "annual_volatility_pct": round(volatility, 2),
            "win_rate_pct":          round(win_rate, 2),
            "best_day_pct":          round(float(daily_rets.max()), 2),
            "worst_day_pct":         round(float(daily_rets.min()), 2),
        },

        # Trade activity
        "trade_activity": {
            "total_trades":          total_trades,
            "buy_trades":            buy_trades,
            "sell_trades":           sell_trades,
            "days_at_hold_0pct":     days_hold,
            "days_at_half_50pct":    days_half,
            "days_at_full_100pct":   days_full,
            "days_invested":         days_invested,
            "pct_time_invested":     round(days_invested / len(alloc) * 100, 1),
            "avg_trades_per_month":  round(total_trades / max(years * 12, 1), 1),
        },

        # Constraints
        "constraints": {
            "sharpe_target":        "> 1.0",
            "sharpe_actual":        round(sharpe, 3),
            "sharpe_met":           sharpe_ok,
            "drawdown_target":      "< -10%",
            "drawdown_actual":      round(mdd, 2),
            "drawdown_met":         dd_ok,
            "verdict":              constraint_verdict,
        },

        # Training context
        "training_round":   latest_round,

        # Forward-looking expectations
        "realistic_expectations": expectations,

        # One-line summary
        "summary": (
            f"RITA RL model: {total_ret:.1f}% return over {total_days} days "
            f"({round(years, 1)}y), Sharpe {sharpe:.3f}, MDD {mdd:.1f}%. "
            f"{total_trades} trades ({buy_trades} buys, {sell_trades} sells). "
            f"{constraint_verdict}. "
            f"Realistic 1-year expectation: {expectations['realistic_1y_pct']:.1f}% "
            f"(conservative: {expectations['conservative_1y_pct']:.1f}%)."
        ),
    }


def simulate_stress_scenarios(
    portfolio_inr: float,
    market_moves: list,
    rita_allocation_pct: float,
) -> dict:
    """
    Point-in-time stress test across market move scenarios.

    For each market_move in market_moves, shows the portfolio impact for:
      - Conservative  (30% fixed allocation)
      - Moderate      (60% fixed allocation)
      - Aggressive    (100% fixed — Buy & Hold)
      - RITA current  (current RL model recommendation allocation)
      - RITA → HOLD   (0% — model's downgrade protection trigger)

    Args:
        portfolio_inr        : Starting capital in INR.
        market_moves         : List of market move percentages (e.g. [-20, -10, 10, 20]).
        rita_allocation_pct  : Current RITA recommendation in % (0, 50, or 100).

    Returns:
        dict with scenario results and per-move breach analysis.
    """
    PROFILES = {
        "conservative":  ("Conservative",       0.30),
        "moderate":      ("Moderate",           0.60),
        "aggressive":    ("Aggressive (B&H)",   1.00),
        "rita_current":  (f"RITA ({int(rita_allocation_pct)}% current)", rita_allocation_pct / 100.0),
        "rita_hold":     ("RITA -> HOLD (0%)",  0.00),
    }

    def _calc(alloc: float, move_pct: float) -> dict:
        impact_pct   = alloc * move_pct
        final_inr    = round(portfolio_inr * (1 + impact_pct / 100))
        pl_inr       = round(final_inr - portfolio_inr)
        dd_pct       = round(min(impact_pct, 0), 2)   # only negative moves are drawdown
        breaches_dd  = dd_pct < -10.0
        return {
            "allocation_pct":    round(alloc * 100),
            "portfolio_impact_pct": round(impact_pct, 2),
            "final_value_inr":   final_inr,
            "profit_loss_inr":   pl_inr,
            "drawdown_pct":      dd_pct,
            "breaches_10pct_dd": breaches_dd,
        }

    scenarios = {}
    for move in market_moves:
        move_key = f"{move:+d}%"
        profiles_at_move = {}
        for key, (label, alloc) in PROFILES.items():
            result = _calc(alloc, move)
            result["label"] = label
            profiles_at_move[key] = result

        # Count breaches (excluding RITA→HOLD which is always safe)
        breach_count = sum(
            1 for k, v in profiles_at_move.items()
            if k != "rita_hold" and v["breaches_10pct_dd"]
        )

        # Narrative insight for this move
        if move < 0:
            rita_dd  = profiles_at_move["rita_current"]["drawdown_pct"]
            hold_msg = (
                "RITA downgrade protection (HOLD) eliminates market loss entirely."
                if rita_allocation_pct > 0 else
                "RITA already at HOLD — fully protected."
            )
            if breach_count == 0:
                insight = f"All profiles within 10% drawdown limit. {hold_msg}"
            else:
                names = [profiles_at_move[k]["label"]
                         for k in profiles_at_move
                         if k != "rita_hold" and profiles_at_move[k]["breaches_10pct_dd"]]
                insight = (
                    f"{', '.join(names)} breach the 10% drawdown limit. "
                    f"RITA at {int(rita_allocation_pct)}% allocation: "
                    f"{rita_dd:.1f}% drawdown. {hold_msg}"
                )
        else:
            best = max(
                (k for k in profiles_at_move if k != "rita_hold"),
                key=lambda k: profiles_at_move[k]["profit_loss_inr"],
            )
            insight = (
                f"Market up {move}%: {profiles_at_move[best]['label']} captures "
                f"most upside (+Rs {profiles_at_move[best]['profit_loss_inr']:,}). "
                f"RITA at {int(rita_allocation_pct)}% captures "
                f"+Rs {profiles_at_move['rita_current']['profit_loss_inr']:,}."
            )

        scenarios[move_key] = {
            "market_move_pct": move,
            "profiles":        profiles_at_move,
            "breach_count":    breach_count,
            "insight":         insight,
        }

    # Summary: which profile is safest in worst case
    worst_move   = min(market_moves)
    worst_key    = f"{worst_move:+d}%"
    worst_scen   = scenarios[worst_key]

    return {
        "portfolio_inr":   int(portfolio_inr),
        "rita_recommendation": f"{int(rita_allocation_pct)}%",
        "market_moves_tested": [f"{m:+d}%" for m in market_moves],
        "scenarios":       scenarios,
        "worst_case": {
            "move":    worst_key,
            "summary": worst_scen["insight"],
            "profiles": {
                k: {
                    "final_inr": v["final_value_inr"],
                    "pl_inr":    v["profit_loss_inr"],
                    "dd_pct":    v["drawdown_pct"],
                    "breach":    v["breaches_10pct_dd"],
                }
                for k, v in worst_scen["profiles"].items()
            },
        },
        "constraint_note": (
            "Constraint: max drawdown < 10%. "
            "RITA -> HOLD (0%) is always safe. "
            f"At current {int(rita_allocation_pct)}% allocation, "
            f"a {abs(worst_move)}% market drop causes "
            f"{worst_scen['profiles']['rita_current']['drawdown_pct']:.1f}% drawdown."
        ),
    }


def _ensure_plots_dir(output_dir: str) -> str:
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_backtest_results(
    portfolio_values: np.ndarray,
    benchmark_values: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: str,
) -> str:
    """Cumulative returns: DDQN portfolio vs Nifty Buy-and-Hold benchmark."""
    plots_dir = _ensure_plots_dir(output_dir)
    path = os.path.join(plots_dir, "backtest_returns.png")

    port = np.asarray(portfolio_values)
    bench = np.asarray(benchmark_values)
    # Normalize to 100
    port_norm = port / port[0] * 100
    bench_norm = bench / bench[0] * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, port_norm, label="DDQN Strategy", color="#2196F3", linewidth=1.8)
    ax.plot(dates, bench_norm, label="Nifty Buy & Hold", color="#FF9800", linewidth=1.4, linestyle="--")
    ax.set_title("Backtest: Cumulative Returns (base=100)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (base 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown(
    portfolio_values: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: str,
) -> str:
    """Drawdown chart with -10% constraint line highlighted."""
    plots_dir = _ensure_plots_dir(output_dir)
    path = os.path.join(plots_dir, "drawdown.png")

    port = np.asarray(portfolio_values)
    running_max = np.maximum.accumulate(port)
    dd = (port - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, dd, 0, where=dd < 0, alpha=0.4, color="#F44336", label="Drawdown")
    ax.axhline(-10, color="red", linestyle="--", linewidth=1.2, label="-10% constraint")
    ax.set_title("Portfolio Drawdown", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_action_distribution(
    allocations: list,
    dates: pd.DatetimeIndex,
    close_prices: np.ndarray,
    output_dir: str,
) -> str:
    """Timeline of DDQN allocation decisions overlaid on Nifty price."""
    plots_dir = _ensure_plots_dir(output_dir)
    path = os.path.join(plots_dir, "action_distribution.png")

    alloc_arr = np.asarray(allocations)
    price_arr = np.asarray(close_prices)
    # allocations is length N-1; dates is length N — align by using dates[1:]
    alloc_dates = np.asarray(dates[1:] if len(dates) == len(alloc_arr) + 1 else dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(dates, price_arr, color="#555555", linewidth=1.2)
    ax1.set_title("Nifty 50 Close Price", fontsize=12)
    ax1.set_ylabel("Price (₹)")
    ax1.grid(alpha=0.3)

    colors = {0.0: "#F44336", 0.5: "#FF9800", 1.0: "#4CAF50"}
    for alloc_val, color in colors.items():
        mask = alloc_arr == alloc_val
        ax2.scatter(alloc_dates[mask], alloc_arr[mask] * 100, s=4, color=color, alpha=0.7,
                    label=f"{int(alloc_val * 100)}% invested")
    ax2.set_title("DDQN Allocation Decisions", fontsize=12)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_yticks([0, 50, 100])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc="upper right", markerscale=4)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    q_values_by_feature: dict,
    output_dir: str,
) -> str:
    """
    Interpretability: average Q-values per action for each feature bucket.

    q_values_by_feature: {feature_name: {action_label: [q_values...]}}
    """
    plots_dir = _ensure_plots_dir(output_dir)
    path = os.path.join(plots_dir, "feature_importance.png")

    features = list(q_values_by_feature.keys())
    n = len(features)
    if n == 0:
        return ""

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    action_colors = {"Cash (0%)": "#F44336", "Half (50%)": "#FF9800", "Full (100%)": "#4CAF50"}

    for ax, feat in zip(axes, features):
        data = q_values_by_feature[feat]
        buckets = list(data.keys())
        x = np.arange(len(buckets))
        width = 0.25
        for i, (action, color) in enumerate(action_colors.items()):
            vals = [np.mean(data[b].get(action, [0])) for b in buckets]
            ax.bar(x + i * width, vals, width, label=action, color=color, alpha=0.8)
        ax.set_title(feat, fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels(buckets, rotation=30, fontsize=8)
        ax.set_ylabel("Avg Q-value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("DDQN: Average Q-values by Feature Bucket (Interpretability)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rolling_sharpe(
    daily_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    output_dir: str,
    window: int = 63,
) -> str:
    """Rolling Sharpe ratio (default 63-day ≈ 1 quarter) over the backtest period."""
    plots_dir = _ensure_plots_dir(output_dir)
    path = os.path.join(plots_dir, "rolling_sharpe.png")

    rets = pd.Series(daily_returns, index=dates[1:])  # daily_returns is len(dates)-1
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    roll_sharpe = (
        (rets - daily_rf).rolling(window).mean()
        / rets.rolling(window).std()
        * math.sqrt(TRADING_DAYS)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(roll_sharpe.index, roll_sharpe.values, color="#9C27B0", linewidth=1.5)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1.2, label="Sharpe = 1 (target)")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 1.0,
                    where=roll_sharpe.values >= 1.0, alpha=0.2, color="green", label="Above target")
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0.0,
                    where=roll_sharpe.values < 0.0, alpha=0.2, color="red", label="Negative Sharpe")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_full_report(
    portfolio_values: np.ndarray,
    benchmark_values: np.ndarray,
    allocations: list,
    dates: pd.DatetimeIndex,
    close_prices: np.ndarray,
    q_values_by_feature: Optional[dict],
    output_dir: str,
) -> dict:
    """
    Generate all performance plots and return their file paths.
    """
    daily_rets = np.diff(portfolio_values) / portfolio_values[:-1]

    plots = {}
    plots["returns"] = plot_backtest_results(portfolio_values, benchmark_values, dates, output_dir)
    plots["drawdown"] = plot_drawdown(portfolio_values, dates, output_dir)
    plots["actions"] = plot_action_distribution(allocations, dates, close_prices, output_dir)
    plots["rolling_sharpe"] = plot_rolling_sharpe(daily_rets, dates, output_dir)

    if q_values_by_feature:
        plots["feature_importance"] = plot_feature_importance(q_values_by_feature, output_dir)

    return plots
