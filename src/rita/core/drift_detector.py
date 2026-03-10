"""
RITA Core — Drift & Health Detector
Monitors model performance stability, data freshness, and pipeline health
across successive training rounds.

Key checks:
  1. Sharpe drift        : Latest Sharpe vs rolling-window mean (warn if Δ > threshold)
  2. Return degradation  : CAGR trend over last N rounds (warn if consistently declining)
  3. Data freshness      : Days since latest date in merged.csv
  4. Pipeline health     : Step failure rate + avg step duration from monitor_log.csv
  5. Constraint breach   : Count of rounds failing Sharpe > 1.0 or MDD < 10%

Status levels:  "ok" | "warn" | "alert"
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

SHARPE_TARGET = 1.0
MDD_LIMIT_PCT = 10.0         # absolute value

DRIFT_WARN_THRESHOLD  = 0.15  # Δ Sharpe > 15% of rolling mean → warn
DRIFT_ALERT_THRESHOLD = 0.30  # Δ Sharpe > 30% → alert

CAGR_DECLINE_ROUNDS = 3       # N consecutive rounds of CAGR decline → warn

DATA_FRESH_WARN_DAYS  = 30    # data older than 30 days → warn
DATA_FRESH_ALERT_DAYS = 90    # data older than 90 days → alert

STEP_FAIL_WARN_RATE   = 0.10  # >10% of recent step runs failed → warn
STEP_FAIL_ALERT_RATE  = 0.25  # >25% failed → alert

ROLLING_WINDOW = 5            # rounds to use for rolling averages


class DriftDetector:
    """
    Analyses training_history.csv and monitor_log.csv to surface performance
    drift, data quality issues, and pipeline health problems.

    Usage:
        detector = DriftDetector(output_dir, csv_path)
        report   = detector.full_report()
        summary  = detector.health_summary(report)   # overall status badge
    """

    def __init__(self, output_dir: str, csv_path: Optional[str] = None):
        self.output_dir = output_dir
        self.csv_path   = csv_path
        self.history_path  = os.path.join(output_dir, "training_history.csv")
        self.monitor_path  = os.path.join(output_dir, "monitor_log.csv")
        self.api_log_path  = os.path.join(output_dir, "api_request_log.csv")

    # ─── Loaders ─────────────────────────────────────────────────────────────

    def _load_history(self) -> pd.DataFrame:
        if os.path.exists(self.history_path):
            try:
                return pd.read_csv(self.history_path)
            except Exception:
                pass
        return pd.DataFrame()

    def _load_monitor(self) -> pd.DataFrame:
        if os.path.exists(self.monitor_path):
            try:
                df = pd.read_csv(self.monitor_path)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                pass
        return pd.DataFrame()

    def _load_api_log(self) -> pd.DataFrame:
        if os.path.exists(self.api_log_path):
            try:
                return pd.read_csv(self.api_log_path)
            except Exception:
                pass
        return pd.DataFrame()

    # ─── Check 1: Sharpe Drift ────────────────────────────────────────────────

    def check_sharpe_drift(self) -> dict:
        """
        Compare latest backtest Sharpe against rolling mean of last N rounds.
        Returns drift_pct (relative deviation), status, and trend series.
        """
        history = self._load_history()
        if history.empty or "backtest_sharpe" not in history.columns:
            return {"status": "ok", "message": "No history yet.", "drift_pct": 0.0, "trend": []}

        sharpe_series = history["backtest_sharpe"].dropna().tolist()
        if len(sharpe_series) < 2:
            return {
                "status": "ok",
                "message": f"Only {len(sharpe_series)} round(s) recorded — need ≥2 for drift.",
                "drift_pct": 0.0,
                "trend": sharpe_series,
            }

        latest = sharpe_series[-1]
        window = sharpe_series[-ROLLING_WINDOW - 1 : -1]  # exclude latest
        rolling_mean = float(np.mean(window)) if window else latest

        drift_pct = abs(latest - rolling_mean) / max(abs(rolling_mean), 1e-6)
        direction = "↓" if latest < rolling_mean else "↑"

        if drift_pct >= DRIFT_ALERT_THRESHOLD:
            status  = "alert"
            message = (f"Sharpe drifted {direction} {drift_pct*100:.1f}% from rolling mean "
                       f"({rolling_mean:.3f} → {latest:.3f})")
        elif drift_pct >= DRIFT_WARN_THRESHOLD:
            status  = "warn"
            message = (f"Sharpe shifted {direction} {drift_pct*100:.1f}% from rolling mean "
                       f"({rolling_mean:.3f} → {latest:.3f})")
        else:
            status  = "ok"
            message = f"Sharpe stable — latest {latest:.3f} vs mean {rolling_mean:.3f}"

        return {
            "status": status,
            "message": message,
            "latest_sharpe": latest,
            "rolling_mean_sharpe": round(rolling_mean, 4),
            "drift_pct": round(drift_pct * 100, 2),
            "direction": direction,
            "trend": sharpe_series,
        }

    # ─── Check 2: Return Degradation ─────────────────────────────────────────

    def check_return_degradation(self) -> dict:
        """
        Detect if CAGR has been consistently declining over the last N rounds.
        """
        history = self._load_history()
        if history.empty or "backtest_cagr_pct" not in history.columns:
            return {"status": "ok", "message": "No history yet.", "trend": []}

        cagr_series = history["backtest_cagr_pct"].dropna().tolist()
        if len(cagr_series) < CAGR_DECLINE_ROUNDS:
            return {
                "status": "ok",
                "message": f"Only {len(cagr_series)} rounds — need ≥{CAGR_DECLINE_ROUNDS} to detect trend.",
                "trend": cagr_series,
            }

        recent = cagr_series[-CAGR_DECLINE_ROUNDS:]
        declining = all(recent[i] < recent[i - 1] for i in range(1, len(recent)))

        if declining:
            drop = recent[0] - recent[-1]
            status  = "warn"
            message = (f"CAGR declining for {CAGR_DECLINE_ROUNDS} consecutive rounds: "
                       f"{recent[0]:.1f}% → {recent[-1]:.1f}% (−{drop:.1f}%)")
        else:
            status  = "ok"
            message = f"No sustained CAGR decline detected (latest {cagr_series[-1]:.1f}%)"

        return {
            "status": status,
            "message": message,
            "recent_cagr": recent,
            "trend": cagr_series,
        }

    # ─── Check 3: Data Freshness ──────────────────────────────────────────────

    def check_data_freshness(self) -> dict:
        """
        Check how old the latest date in the Nifty CSV is.
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            return {"status": "warn", "message": "CSV path not configured or file missing.", "days_old": None}

        try:
            # Read just the last few rows for performance
            df = pd.read_csv(self.csv_path, usecols=[0], header=0)
            date_col = df.columns[0]
            latest_str = df[date_col].dropna().iloc[-1]
            latest_date = pd.to_datetime(latest_str, dayfirst=False).date()
            today = datetime.now().date()
            days_old = (today - latest_date).days
        except Exception as exc:
            return {"status": "warn", "message": f"Could not parse CSV date: {exc}", "days_old": None}

        if days_old >= DATA_FRESH_ALERT_DAYS:
            status  = "alert"
            message = f"Market data is {days_old} days old (latest: {latest_date}) — refresh recommended"
        elif days_old >= DATA_FRESH_WARN_DAYS:
            status  = "warn"
            message = f"Market data is {days_old} days old (latest: {latest_date})"
        else:
            status  = "ok"
            message = f"Market data is current — {days_old} day(s) old (latest: {latest_date})"

        return {
            "status": status,
            "message": message,
            "latest_date": str(latest_date),
            "days_old": days_old,
        }

    # ─── Check 4: Pipeline Health ─────────────────────────────────────────────

    def check_pipeline_health(self) -> dict:
        """
        Analyse recent step failure rate and average step durations.
        Uses the last 50 completed step rows from monitor_log.csv.
        """
        monitor = self._load_monitor()
        if monitor.empty or "status" not in monitor.columns:
            return {"status": "ok", "message": "No pipeline runs recorded yet.", "step_stats": {}}

        # Only look at rows that have ended (not in_progress)
        ended = monitor[monitor["status"].isin(["completed", "failed"])].tail(50)
        if ended.empty:
            return {"status": "ok", "message": "No completed/failed steps yet.", "step_stats": {}}

        total = len(ended)
        failed = (ended["status"] == "failed").sum()
        fail_rate = failed / total if total > 0 else 0.0

        # Per-step average duration
        step_stats: dict = {}
        if "step_num" in ended.columns and "duration_secs" in ended.columns:
            dur_df = ended.dropna(subset=["duration_secs"])
            dur_df = dur_df.copy()
            dur_df["duration_secs"] = pd.to_numeric(dur_df["duration_secs"], errors="coerce")
            for snum, grp in dur_df.groupby("step_num"):
                step_stats[int(snum)] = {
                    "avg_duration_secs": round(float(grp["duration_secs"].mean()), 1),
                    "max_duration_secs": round(float(grp["duration_secs"].max()), 1),
                    "runs": int(len(grp)),
                    "failures": int((grp.get("status", pd.Series()) == "failed").sum()),
                }

        # Last N runs summary (most recent full runs)
        recent_runs = _extract_recent_runs(monitor)

        if fail_rate >= STEP_FAIL_ALERT_RATE:
            status  = "alert"
            message = f"High step failure rate: {failed}/{total} ({fail_rate*100:.0f}%) in last 50 entries"
        elif fail_rate >= STEP_FAIL_WARN_RATE:
            status  = "warn"
            message = f"Elevated failure rate: {failed}/{total} ({fail_rate*100:.0f}%) in last 50 entries"
        else:
            status  = "ok"
            message = f"Pipeline healthy — {failed}/{total} failures ({fail_rate*100:.0f}%) in last 50 entries"

        return {
            "status": status,
            "message": message,
            "total_logged": total,
            "failed_steps": int(failed),
            "fail_rate_pct": round(fail_rate * 100, 1),
            "step_stats": step_stats,
            "recent_runs": recent_runs,
        }

    # ─── Check 5: Constraint Breach Rate ──────────────────────────────────────

    def check_constraint_breach(self) -> dict:
        """
        Count how many recorded rounds failed the Sharpe or MDD constraint.
        """
        history = self._load_history()
        if history.empty:
            return {"status": "ok", "message": "No history yet.", "breach_rounds": []}

        breaches = []
        for _, row in history.iterrows():
            sharpe_ok = float(row.get("backtest_sharpe", 0)) >= SHARPE_TARGET
            mdd_ok    = abs(float(row.get("backtest_mdd_pct", 0))) <= MDD_LIMIT_PCT
            if not (sharpe_ok and mdd_ok):
                breaches.append({
                    "round": int(row.get("round", 0)),
                    "sharpe": round(float(row.get("backtest_sharpe", 0)), 3),
                    "mdd_pct": round(float(row.get("backtest_mdd_pct", 0)), 2),
                })

        total = len(history)
        breach_count = len(breaches)
        breach_rate  = breach_count / total if total > 0 else 0.0

        if breach_rate >= 0.5:
            status  = "warn"
            message = f"{breach_count}/{total} rounds failed constraints ({breach_rate*100:.0f}%)"
        elif breach_count > 0:
            status  = "ok"
            message = f"{breach_count}/{total} rounds failed constraints — latest round OK"
        else:
            status  = "ok"
            message = f"All {total} rounds met constraints ✓"

        return {
            "status": status,
            "message": message,
            "total_rounds": total,
            "breach_count": breach_count,
            "breach_rate_pct": round(breach_rate * 100, 1),
            "breach_rounds": breaches,
        }

    # ─── Full Report ──────────────────────────────────────────────────────────

    def full_report(self) -> dict:
        """
        Run all checks and return a structured dict suitable for display.
        """
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sharpe_drift":    self.check_sharpe_drift(),
            "return_degradation": self.check_return_degradation(),
            "data_freshness":  self.check_data_freshness(),
            "pipeline_health": self.check_pipeline_health(),
            "constraint_breach": self.check_constraint_breach(),
        }

    # ─── Health Summary ───────────────────────────────────────────────────────

    def health_summary(self, report: Optional[dict] = None) -> dict:
        """
        Roll up all check statuses into a single badge.

        Returns:
            {
              "overall": "ok" | "warn" | "alert",
              "checks":  dict of check_name → status
            }
        """
        if report is None:
            report = self.full_report()

        check_keys = [
            "sharpe_drift", "return_degradation",
            "data_freshness", "pipeline_health", "constraint_breach",
        ]
        statuses = {k: report[k]["status"] for k in check_keys if k in report}

        if any(s == "alert" for s in statuses.values()):
            overall = "alert"
        elif any(s == "warn" for s in statuses.values()):
            overall = "warn"
        else:
            overall = "ok"

        return {"overall": overall, "checks": statuses}

    # ─── API Request Stats (for DevOps tab) ───────────────────────────────────

    def api_request_stats(self) -> dict:
        """
        Aggregate API request log into summary stats.
        Returns counts, avg latency, error rate, and per-endpoint breakdown.
        """
        log = self._load_api_log()
        if log.empty:
            return {"total_requests": 0, "error_count": 0, "endpoints": {}}

        total = len(log)
        errors = int((log.get("status_code", pd.Series(dtype=int)) >= 400).sum()) if "status_code" in log else 0
        avg_latency = float(log["duration_ms"].mean()) if "duration_ms" in log else 0.0

        endpoints: dict = {}
        if "path" in log.columns and "duration_ms" in log.columns:
            for path, grp in log.groupby("path"):
                endpoints[path] = {
                    "count": int(len(grp)),
                    "avg_ms": round(float(grp["duration_ms"].mean()), 1),
                    "errors": int((grp.get("status_code", pd.Series(dtype=int)) >= 400).sum()),
                }

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate_pct": round(errors / total * 100, 1) if total > 0 else 0.0,
            "avg_latency_ms": round(avg_latency, 1),
            "endpoints": endpoints,
            "recent": log.tail(20).to_dict(orient="records") if not log.empty else [],
        }


# ─── Helper ───────────────────────────────────────────────────────────────────

def _extract_recent_runs(monitor_df: pd.DataFrame, n: int = 10) -> list:
    """
    Extract the most recent N complete pipeline runs (steps 1-8 cycles)
    from monitor_log.csv.  Returns a list of dicts with run_date, total_secs,
    completed_steps, and failed_steps.
    """
    if monitor_df.empty or "started_at" not in monitor_df.columns:
        return []

    df = monitor_df.copy()
    df = df[df["status"].isin(["completed", "failed"])].copy()
    if df.empty:
        return []

    try:
        df["started_at_dt"] = pd.to_datetime(df["started_at"], errors="coerce")
    except Exception:
        return []

    # Group by calendar date (approximate run grouping)
    df["run_date"] = df["started_at_dt"].dt.date

    runs = []
    for run_date, grp in df.groupby("run_date"):
        dur_vals = pd.to_numeric(grp.get("duration_secs", pd.Series()), errors="coerce")
        total_secs = round(float(dur_vals.sum()), 1)
        runs.append({
            "run_date": str(run_date),
            "completed_steps": int((grp["status"] == "completed").sum()),
            "failed_steps":    int((grp["status"] == "failed").sum()),
            "total_secs": total_secs,
        })

    runs.sort(key=lambda x: x["run_date"], reverse=True)
    return runs[:n]
