"""
RITA Core — Risk Engine
Computes trade-level risk metrics across all three phases (Train / Validation / Backtest).

Metrics per day:
  rolling_vol_20d     : 20-day annualised market volatility (%)
  portfolio_var_95    : 1-day 95% Historical VaR scaled by allocation (% of portfolio)
  portfolio_cvar_95   : Expected Shortfall at 95%, scaled by allocation (% of portfolio)
  current_drawdown_pct: Drawdown from portfolio peak (%)
  drawdown_budget_pct : % of the 10% MDD limit consumed (0–100)
  regime              : Bull / Bear / Sideways  (from trend_score)
  model_confidence    : Q-value spread max−min  (agent conviction, NaN if unavailable)
  position_risk_pct   : allocation × daily_vol  (% of portfolio at risk per day)

Starting state: 100% cash (₹10,00,000) → VaR = 0%, position_risk = 0%
"""

import json
import os

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
MDD_LIMIT = 0.10        # 10% max drawdown constraint
VAR_WINDOW = 252        # 1-year lookback for historical VaR
VOL_WINDOW = 20         # 20-day rolling volatility
INITIAL_CAPITAL = 1_000_000   # ₹10L starting portfolio

REGIME_BULL_THRESHOLD = 0.3
REGIME_BEAR_THRESHOLD = -0.3


class RiskEngine:
    """
    Computes per-day and per-trade risk metrics for RITA's 8-step workflow.

    Usage:
        engine = RiskEngine()
        train_tl  = engine.compute_risk_timeline(train_episode, feat_df, nifty_rets, "Train")
        val_tl    = engine.compute_risk_timeline(val_episode,   feat_df, nifty_rets, "Validation")
        bt_tl     = engine.compute_risk_timeline(bt_episode,    feat_df, nifty_rets, "Backtest")
        combined  = engine.combine_phases([train_tl, val_tl, bt_tl])
        trades    = engine.compute_trade_events(combined)
        summary   = engine.compute_risk_summary(combined, trades)
        engine.save(combined, trades, summary, output_dir)
    """

    def __init__(
        self,
        mdd_limit: float = MDD_LIMIT,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.mdd_limit = mdd_limit
        self.initial_capital = initial_capital

    # ─── Core computation ─────────────────────────────────────────────────────

    def compute_risk_timeline(
        self,
        episode_result: dict,
        feature_df: pd.DataFrame,
        nifty_returns: pd.Series,
        phase: str = "Backtest",
    ) -> pd.DataFrame:
        """
        Compute per-day risk metrics for one phase.

        Args:
            episode_result : dict from run_episode() — must contain:
                             dates, portfolio_values, allocations.
                             Optionally: q_confidence_series.
            feature_df     : feature-enriched price data for this period
                             (must contain trend_score column).
            nifty_returns  : full historical Nifty daily return series
                             (used for rolling VaR / CVaR lookback; must
                              include at least VAR_WINDOW days before phase start).
            phase          : "Train" | "Validation" | "Backtest"

        Returns:
            DataFrame indexed by date with all risk metric columns.
        """
        dates = pd.DatetimeIndex(episode_result["dates"])
        portfolio_values = np.array(episode_result["portfolio_values"], dtype=float)
        allocations = list(episode_result["allocations"])
        q_conf_raw = episode_result.get("q_confidence_series", [])

        n = len(dates)

        # Align allocation series to date index.
        # Action at dates[i] earns the return at dates[i+1], but we show the
        # allocation decided at dates[i] as the "active allocation" for that day.
        if allocations:
            alloc_series = list(allocations) + [allocations[-1]]
        else:
            alloc_series = [0.0] * n

        if q_conf_raw:
            q_conf_series = list(q_conf_raw) + [q_conf_raw[-1]]
        else:
            q_conf_series = [float("nan")] * n

        # ── Pre-compute rolling metrics on the full Nifty return history ──────
        # Using vectorised pandas operations — O(N log N), computed once.
        rolling_vol = nifty_returns.rolling(VOL_WINDOW).std() * np.sqrt(252) * 100

        # Historical VaR 95% — quantile is vectorised in pandas
        rolling_var95 = nifty_returns.rolling(VAR_WINDOW).quantile(0.05) * -100

        # CVaR (Expected Shortfall) — requires apply but computed once on full series
        def _cvar(x: np.ndarray) -> float:
            thresh = np.percentile(x, 5)
            tail = x[x <= thresh]
            return float(-tail.mean() * 100) if len(tail) > 0 else float(-thresh * 100)

        rolling_cvar95 = nifty_returns.rolling(VAR_WINDOW).apply(_cvar, raw=True)

        # Reindex rolling series to the episode dates (forward-fill gaps)
        vol_vals = rolling_vol.reindex(dates, method="ffill").values
        var_vals = rolling_var95.reindex(dates, method="ffill").fillna(0.0).values
        cvar_vals = rolling_cvar95.reindex(dates, method="ffill").fillna(0.0).values

        # ── Trend score for regime detection ──────────────────────────────────
        if "trend_score" in feature_df.columns:
            ts_vals = (
                feature_df["trend_score"]
                .reindex(dates, method="ffill")
                .fillna(0.0)
                .values
            )
        else:
            ts_vals = np.zeros(n)

        # ── Portfolio drawdown from running peak ──────────────────────────────
        running_peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_peak) / running_peak * 100  # negative

        # ── Build per-day rows ────────────────────────────────────────────────
        rows = []
        for i in range(n):
            alloc = float(alloc_series[i])
            pv = float(portfolio_values[i])
            vol = float(vol_vals[i]) if not np.isnan(vol_vals[i]) else 0.0
            var_mkt = float(var_vals[i])
            cvar_mkt = float(cvar_vals[i])
            ts = float(ts_vals[i])

            # Scale market VaR by allocation → portfolio VaR
            port_var = alloc * var_mkt
            port_cvar = alloc * cvar_mkt

            # Daily position risk: allocation × (annualised_vol / √252)
            daily_vol = vol / np.sqrt(252) if vol > 0 else 0.0
            position_risk = alloc * daily_vol

            # Drawdown budget
            dd_pct = float(drawdown[i])
            budget_pct = min(abs(dd_pct) / (self.mdd_limit * 100) * 100, 100.0)

            # Regime
            if ts > REGIME_BULL_THRESHOLD:
                regime = "Bull"
            elif ts < REGIME_BEAR_THRESHOLD:
                regime = "Bear"
            else:
                regime = "Sideways"

            # INR values
            port_inr = pv * self.initial_capital
            inr_at_risk = port_var / 100 * port_inr

            qc = q_conf_series[i]
            rows.append(
                {
                    "date": dates[i],
                    "phase": phase,
                    "allocation": alloc,
                    "portfolio_value_norm": pv,
                    "portfolio_value_inr": port_inr,
                    "rolling_vol_20d": vol,
                    "market_var_95": var_mkt,
                    "portfolio_var_95": port_var,
                    "portfolio_cvar_95": port_cvar,
                    "current_drawdown_pct": dd_pct,
                    "drawdown_budget_pct": budget_pct,
                    "position_risk_pct": position_risk,
                    "trend_score": ts,
                    "regime": regime,
                    "model_confidence": float(qc) if (qc is not None and not (isinstance(qc, float) and np.isnan(qc))) else None,
                    "inr_at_risk": inr_at_risk,
                }
            )

        return pd.DataFrame(rows).set_index("date")

    # ─── Trade events ─────────────────────────────────────────────────────────

    def compute_trade_events(self, combined: pd.DataFrame) -> pd.DataFrame:
        """
        Extract days when the allocation changes — these are trade events.
        Computes ΔVaR and a risk_action label for each trade.

        Returns a flat DataFrame (not indexed by date).
        """
        df = combined.reset_index().copy()
        df["prev_allocation"] = df["allocation"].shift(1)
        df["prev_var"] = df["portfolio_var_95"].shift(1)

        trades = df[
            df["prev_allocation"].notna()
            & (df["allocation"] != df["prev_allocation"])
        ].copy()

        trades["delta_allocation"] = trades["allocation"] - trades["prev_allocation"]
        trades["delta_var"] = trades["portfolio_var_95"] - trades["prev_var"]
        trades["risk_action"] = trades["delta_var"].apply(
            lambda x: "Increased" if x > 0.001 else ("Reduced" if x < -0.001 else "Neutral")
        )

        keep = [
            "date", "phase", "prev_allocation", "allocation", "delta_allocation",
            "portfolio_var_95", "delta_var", "risk_action",
            "current_drawdown_pct", "drawdown_budget_pct",
            "regime", "model_confidence",
            "portfolio_value_inr", "rolling_vol_20d",
        ]
        return trades[keep].reset_index(drop=True)

    # ─── Summary statistics ────────────────────────────────────────────────────

    def compute_risk_summary(
        self, combined: pd.DataFrame, trades: pd.DataFrame
    ) -> dict:
        """
        Aggregate risk statistics: overall totals + per-phase breakdown.
        """
        invested = combined[combined["allocation"] > 0]
        total_return = float(combined["portfolio_value_norm"].iloc[-1]) - 1.0
        avg_var = float(invested["portfolio_var_95"].mean()) if len(invested) > 0 else 0.0

        peak_var_idx = combined["portfolio_var_95"].idxmax()

        summary: dict = {
            "starting_var_pct": 0.0,
            "avg_var_when_invested": round(avg_var, 3),
            "peak_var_pct": round(float(combined["portfolio_var_95"].max()), 3),
            "peak_var_date": str(peak_var_idx.date()) if hasattr(peak_var_idx, "date") else str(peak_var_idx),
            "avg_vol_20d": round(float(combined["rolling_vol_20d"].mean()), 2),
            "peak_vol_20d": round(float(combined["rolling_vol_20d"].max()), 2),
            "max_drawdown_budget_pct": round(float(combined["drawdown_budget_pct"].max()), 1),
            "budget_breach_days": int((combined["drawdown_budget_pct"] >= 100).sum()),
            "total_trades": len(trades),
            "risk_increasing_trades": int((trades["delta_var"] > 0.001).sum()),
            "risk_reducing_trades": int((trades["delta_var"] < -0.001).sum()),
            "risk_adjusted_return": round(total_return / max(avg_var / 100, 0.001), 3),
        }

        per_phase: dict = {}
        for phase in combined["phase"].unique():
            ph = combined[combined["phase"] == phase]
            pt = trades[trades["phase"] == phase]
            per_phase[phase] = {
                "avg_vol_pct": round(float(ph["rolling_vol_20d"].mean()), 2),
                "avg_var_pct": round(float(ph["portfolio_var_95"].mean()), 2),
                "max_drawdown_pct": round(float(ph["current_drawdown_pct"].min()), 2),
                "max_budget_used_pct": round(float(ph["drawdown_budget_pct"].max()), 1),
                "trade_count": len(pt),
                "days": len(ph),
            }
        summary["per_phase"] = per_phase
        return summary

    # ─── Combine phases ───────────────────────────────────────────────────────

    def combine_phases(self, timelines: list) -> pd.DataFrame:
        """Concatenate phase DataFrames into one continuous timeline, sorted by date."""
        return pd.concat(timelines).sort_index()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(
        self,
        combined: pd.DataFrame,
        trades: pd.DataFrame,
        summary: dict,
        output_dir: str,
    ) -> None:
        """Save risk data to output_dir as CSV + JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        combined.to_csv(os.path.join(output_dir, "risk_timeline.csv"))
        trades.to_csv(os.path.join(output_dir, "risk_trade_events.csv"), index=False)
        with open(os.path.join(output_dir, "risk_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    @staticmethod
    def load(output_dir: str) -> tuple:
        """
        Load saved risk data from output_dir.

        Returns:
            (combined_df, trades_df, summary_dict)  on success
            (None, None, None)                      if files not found
        """
        try:
            combined = pd.read_csv(
                os.path.join(output_dir, "risk_timeline.csv"),
                index_col="date",
                parse_dates=True,
            )
            trades = pd.read_csv(
                os.path.join(output_dir, "risk_trade_events.csv"),
                parse_dates=["date"],
            )
            with open(os.path.join(output_dir, "risk_summary.json")) as fh:
                summary = json.load(fh)
            return combined, trades, summary
        except (FileNotFoundError, KeyError, ValueError):
            return None, None, None
