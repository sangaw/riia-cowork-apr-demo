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
