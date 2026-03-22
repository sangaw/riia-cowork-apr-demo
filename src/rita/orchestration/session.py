"""
RITA Orchestration — Session Manager
In-memory state store with CSV persistence.
No database — all state is stored as flat CSV files.
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from typing import Any


def _atomic_csv(path: str, header: list, rows: list) -> None:
    """Write a CSV atomically: write to a temp file, then rename into place.

    Prevents partial-write corruption if the process is killed mid-save.
    The temp file is created in the same directory so os.replace() is always
    an atomic rename on the same filesystem.
    """
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix=".rita_")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SessionManager:
    """
    Manages the RITA workflow session state.

    Keys:
        goal              — dict from set_goal()
        research          — dict from analyze_market()
        strategy          — dict from design_strategy()
        model_path        — str path to saved DDQN model zip
        training_metrics  — dict from train_agent()
        validation_metrics— dict from validate_agent()
        simulation_period — dict {start, end}
        backtest_results  — dict from run_episode()
        updated_goal      — dict from update_goal_from_results()
    """

    def __init__(self, output_dir: str = "./rita_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._state: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def get(self, key: str, default=None) -> Any:
        return self._state.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._state

    def clear(self) -> None:
        self._state = {}

    def save(self) -> None:
        """Persist session state to CSV files in output_dir (all writes are atomic)."""
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().isoformat()

        # session_state.csv — flat key/value snapshot
        state_rows = [
            [k, json.dumps(v, default=str), ts]
            for k, v in self._state.items()
            if k not in ("backtest_results",)   # large dicts handled separately
        ]
        _atomic_csv(
            os.path.join(self.output_dir, "session_state.csv"),
            ["key", "value", "saved_at"],
            state_rows,
        )

        # backtest_daily.csv — daily time-series from backtest
        backtest = self._state.get("backtest_results", {})
        if backtest:
            dates       = backtest.get("dates", [])
            port_vals   = backtest.get("portfolio_values", [])
            bench_vals  = backtest.get("benchmark_values", [])
            allocs      = backtest.get("allocations", [])
            close_prices = backtest.get("close_prices", [])
            daily_rows = [
                [
                    str(date)[:10],
                    round(port_vals[i], 6) if i < len(port_vals) else "",
                    round(bench_vals[i], 6) if i < len(bench_vals) else "",
                    allocs[i - 1] if i > 0 and i - 1 < len(allocs) else "",
                    round(close_prices[i], 2) if i < len(close_prices) else "",
                ]
                for i, date in enumerate(dates)
            ]
            _atomic_csv(
                os.path.join(self.output_dir, "backtest_daily.csv"),
                ["date", "portfolio_value", "benchmark_value", "allocation", "close_price"],
                daily_rows,
            )

        # performance_summary.csv
        perf = backtest.get("performance", {}) if backtest else {}
        if perf:
            _atomic_csv(
                os.path.join(self.output_dir, "performance_summary.csv"),
                ["metric", "value"],
                [[k, v] for k, v in perf.items()],
            )

        # goal_history.csv
        goal = self._state.get("goal")
        updated_goal = self._state.get("updated_goal")
        if goal or updated_goal:
            goal_rows = []
            if goal:
                goal_rows += [["original", k, json.dumps(v, default=str)] for k, v in goal.items()]
            if updated_goal:
                goal_rows += [["updated", k, json.dumps(v, default=str)] for k, v in updated_goal.items()]
            _atomic_csv(
                os.path.join(self.output_dir, "goal_history.csv"),
                ["stage", "key", "value"],
                goal_rows,
            )

    def load(self) -> bool:
        """Restore session state from CSV files. Returns True if data was found."""
        state_path = os.path.join(self.output_dir, "session_state.csv")
        if not os.path.exists(state_path):
            return False

        with open(state_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self._state[row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return True

    def get_progress_summary(self) -> dict:
        """Return which steps have been completed."""
        steps = {
            "step1_goal_set": self.has("goal"),
            "step2_market_analyzed": self.has("research"),
            "step3_strategy_designed": self.has("strategy"),
            "step4_model_trained": self.has("model_path"),
            "step5_period_set": self.has("simulation_period"),
            "step6_backtest_run": self.has("backtest_results"),
            "step7_results_ready": self.has("backtest_results"),
            "step8_goal_updated": self.has("updated_goal"),
        }
        completed = sum(steps.values())
        return {
            "steps_completed": completed,
            "total_steps": 8,
            "pct_complete": round(completed / 8 * 100),
            "steps": steps,
        }
