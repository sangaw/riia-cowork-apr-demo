"""
RITA Orchestration — Phase Monitor
Tracks progress, timing, and status of each workflow step.
Outputs a structured log to rita_output/monitor_log.csv.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional


STEP_NAMES = {
    1: "Set Financial Goal",
    2: "Analyze Market Conditions",
    3: "Design Strategy",
    4: "Train DDQN Model",
    5: "Set Simulation Period",
    6: "Run Backtest",
    7: "Get Results & Plots",
    8: "Update Financial Goal",
}


class PhaseMonitor:
    """
    Tracks step timing, status, and writes a running CSV log.
    """

    def __init__(self, output_dir: str = "./rita_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "monitor_log.csv")
        self._steps: dict[int, dict] = {}
        self._current_step: Optional[int] = None
        self._step_start: Optional[float] = None
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step_num", "step_name", "status",
                    "started_at", "ended_at", "duration_secs", "summary"
                ])

    def start_step(self, step_num: int) -> None:
        step_name = STEP_NAMES.get(step_num, f"Step {step_num}")
        self._current_step = step_num
        self._step_start = time.time()
        self._steps[step_num] = {
            "name": step_name,
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "duration_secs": None,
            "summary": {},
        }
        self._append_log(step_num, "in_progress", None, None, {})

    def end_step(self, step_num: int, result_summary: dict, status: str = "completed") -> None:
        if step_num not in self._steps:
            return
        duration = round(time.time() - (self._step_start or time.time()), 2)
        ended_at = datetime.now().isoformat()
        self._steps[step_num].update({
            "status": status,
            "ended_at": ended_at,
            "duration_secs": duration,
            "summary": result_summary,
        })
        self._append_log(step_num, status, ended_at, duration, result_summary)
        self._current_step = None

    def fail_step(self, step_num: int, error: str) -> None:
        self.end_step(step_num, {"error": error}, status="failed")

    def _append_log(self, step_num, status, ended_at, duration, summary):
        step_name = STEP_NAMES.get(step_num, f"Step {step_num}")
        started_at = self._steps.get(step_num, {}).get("started_at", "")
        # Keep summary as a brief string
        summary_str = "; ".join(
            f"{k}={v}" for k, v in summary.items()
            if not isinstance(v, (dict, list))
        )[:200]
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step_num, step_name, status,
                started_at, ended_at or "", duration or "",
                summary_str
            ])

    def get_progress_report(self) -> dict:
        """Return a structured progress report for all steps."""
        completed = [s for s in self._steps.values() if s["status"] == "completed"]
        failed = [s for s in self._steps.values() if s["status"] == "failed"]

        step_statuses = {}
        for i in range(1, 9):
            if i in self._steps:
                step_statuses[STEP_NAMES[i]] = self._steps[i]["status"]
            else:
                step_statuses[STEP_NAMES[i]] = "pending"

        return {
            "steps_completed": len(completed),
            "steps_failed": len(failed),
            "total_steps": 8,
            "pct_complete": round(len(completed) / 8 * 100),
            "current_step": STEP_NAMES.get(self._current_step, "None"),
            "step_statuses": step_statuses,
            "timings_secs": {
                STEP_NAMES[k]: v["duration_secs"]
                for k, v in self._steps.items()
                if v["duration_secs"] is not None
            },
        }

    def print_progress(self) -> None:
        """Print a human-readable progress bar to stdout."""
        report = self.get_progress_report()
        done = report["steps_completed"]
        total = report["total_steps"]
        bar = "█" * done + "░" * (total - done)
        print(f"\nRITA Progress: [{bar}] {done}/{total} steps ({report['pct_complete']}%)")
        for name, status in report["step_statuses"].items():
            icon = {"completed": "✓", "in_progress": "▶", "failed": "✗", "pending": "·"}.get(status, "?")
            print(f"  {icon} {name} — {status}")
        print()
