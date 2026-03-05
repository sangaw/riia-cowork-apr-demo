"""
RITA Orchestration — Session Manager
In-memory state store with CSV persistence.
No database — all state is stored as flat CSV files.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any


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
        """Persist session state to CSV files in output_dir."""
        os.makedirs(self.output_dir, exist_ok=True)

        # session_state.csv — flat key/value snapshot
        state_path = os.path.join(self.output_dir, "session_state.csv")
        with open(state_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value", "saved_at"])
            ts = datetime.now().isoformat()
            for k, v in self._state.items():
                if k not in ("backtest_results",):  # large dicts handled separately
                    writer.writerow([k, json.dumps(v, default=str), ts])

        # backtest_daily.csv — daily time-series from backtest
        backtest = self._state.get("backtest_results", {})
        if backtest:
            daily_path = os.path.join(self.output_dir, "backtest_daily.csv")
            dates = backtest.get("dates", [])
            port_vals = backtest.get("portfolio_values", [])
            bench_vals = backtest.get("benchmark_values", [])
            allocs = backtest.get("allocations", [])
            close_prices = backtest.get("close_prices", [])

            with open(daily_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "portfolio_value", "benchmark_value",
                                 "allocation", "close_price"])
                for i, date in enumerate(dates):
                    alloc = allocs[i - 1] if i > 0 and i - 1 < len(allocs) else ""
                    writer.writerow([
                        str(date)[:10],
                        round(port_vals[i], 6) if i < len(port_vals) else "",
                        round(bench_vals[i], 6) if i < len(bench_vals) else "",
                        alloc,
                        round(close_prices[i], 2) if i < len(close_prices) else "",
                    ])

        # performance_summary.csv
        perf = backtest.get("performance", {})
        if perf:
            perf_path = os.path.join(self.output_dir, "performance_summary.csv")
            with open(perf_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in perf.items():
                    writer.writerow([k, v])

        # goal_history.csv
        goal = self._state.get("goal")
        updated_goal = self._state.get("updated_goal")
        if goal or updated_goal:
            goal_path = os.path.join(self.output_dir, "goal_history.csv")
            with open(goal_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["stage", "key", "value"])
                if goal:
                    for k, v in goal.items():
                        writer.writerow(["original", k, json.dumps(v, default=str)])
                if updated_goal:
                    for k, v in updated_goal.items():
                        writer.writerow(["updated", k, json.dumps(v, default=str)])

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
