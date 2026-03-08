"""
RITA Core — Training Round Tracker
Records metrics for each training/evaluation run so model improvement can be
visualised over successive fine-tuning cycles in the Training Progress tab.

Output file: <output_dir>/training_history.csv

Each row represents one complete pipeline run (Steps 1-8).
Round numbers are assigned sequentially; re-runs with a loaded (not retrained)
model are recorded separately so the analyst can compare evaluation stability.
"""

import os
from datetime import datetime

import pandas as pd

HISTORY_FILE = "training_history.csv"

COLUMNS = [
    "round",
    "timestamp",
    "timesteps",
    "source",                  # "trained" | "loaded_existing"
    # Validation metrics (2023-2024)
    "val_sharpe",
    "val_mdd_pct",
    "val_cagr_pct",
    "val_constraints_met",
    # Backtest metrics (2025+)
    "backtest_sharpe",
    "backtest_mdd_pct",
    "backtest_return_pct",
    "backtest_cagr_pct",
    "backtest_constraints_met",
    "notes",
]


class TrainingTracker:
    """
    Appends one row per pipeline run to training_history.csv.

    Usage:
        tracker = TrainingTracker(output_dir)
        round_num = tracker.record_round(training_metrics, val_metrics, backtest_metrics)
        history   = tracker.load_history()   # → pd.DataFrame
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history_path = os.path.join(output_dir, HISTORY_FILE)
        os.makedirs(output_dir, exist_ok=True)

    # ─── Write ────────────────────────────────────────────────────────────────

    def record_round(
        self,
        training_metrics: dict,
        val_metrics: dict,
        backtest_metrics: dict,
        notes: str = "",
    ) -> int:
        """
        Append a new round to the history CSV.

        Args:
            training_metrics : dict from step4 result (keys: timesteps_trained, source)
            val_metrics      : dict from validate_agent() (keys: sharpe_ratio, max_drawdown_pct, …)
            backtest_metrics : dict from backtest performance (keys: sharpe_ratio, max_drawdown_pct, …)
            notes            : free-text label (e.g., "lr=1e-4, 200k steps")

        Returns:
            The round number just recorded (1-based).
        """
        history = self.load_history()
        round_num = len(history) + 1

        def _f(d: dict, key: str, default=0.0):
            return round(float(d.get(key, default)), 4)

        row = {
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timesteps": int(training_metrics.get("timesteps_trained", 0)),
            "source": training_metrics.get("source", "trained"),
            "val_sharpe": _f(val_metrics, "sharpe_ratio"),
            "val_mdd_pct": _f(val_metrics, "max_drawdown_pct"),
            "val_cagr_pct": _f(val_metrics, "portfolio_cagr_pct"),
            "val_constraints_met": bool(val_metrics.get("constraints_met", False)),
            "backtest_sharpe": _f(backtest_metrics, "sharpe_ratio"),
            "backtest_mdd_pct": _f(backtest_metrics, "max_drawdown_pct"),
            "backtest_return_pct": _f(backtest_metrics, "portfolio_total_return_pct"),
            "backtest_cagr_pct": _f(backtest_metrics, "portfolio_cagr_pct"),
            "backtest_constraints_met": bool(backtest_metrics.get("constraints_met", False)),
            "notes": notes,
        }

        new_df = pd.DataFrame([row], columns=COLUMNS)
        if history.empty:
            updated = new_df
        else:
            updated = pd.concat([history, new_df], ignore_index=True)

        updated.to_csv(self.history_path, index=False)
        return round_num

    # ─── Read ─────────────────────────────────────────────────────────────────

    def load_history(self) -> pd.DataFrame:
        """Return the full training history DataFrame (empty if no file yet)."""
        if os.path.exists(self.history_path):
            try:
                return pd.read_csv(self.history_path)
            except Exception:
                pass
        return pd.DataFrame(columns=COLUMNS)

    def get_round_count(self) -> int:
        return len(self.load_history())

    def get_latest_round(self) -> dict | None:
        history = self.load_history()
        return history.iloc[-1].to_dict() if not history.empty else None
