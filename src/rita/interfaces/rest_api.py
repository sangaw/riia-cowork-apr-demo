"""
RITA Interface — FastAPI REST API (Phase 3)
Exposes the 8-step workflow as HTTP endpoints.

Run with:
    python run_api.py
    uvicorn rita.interfaces.rest_api:app --reload --port 8000

Docs auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import BACKTEST_START
from rita.core.drift_detector import DriftDetector
from rita.config import (
    NIFTY_CSV_PATH as CSV_PATH,
    OUTPUT_DIR,
    INPUT_DIR,
    PORTFOLIO_API_KEY,
)

# ─── Configuration ────────────────────────────────────────────────────────────

API_LOG_PATH = os.path.join(OUTPUT_DIR, "api_request_log.csv")

# Path where prepare_data() writes the extended merged CSV
_PREPARED_CSV = os.path.join(OUTPUT_DIR, "nifty_merged.csv")


def _get_active_csv() -> str:
    """Return the extended CSV if prepare_data() has been run, else the base CSV."""
    return _PREPARED_CSV if os.path.exists(_PREPARED_CSV) else CSV_PATH

# ─── Portfolio auth guard ─────────────────────────────────────────────────────

def _require_portfolio_key(x_api_key: str = Header(default="")) -> None:
    """Dependency: reject requests that don't carry the configured API key.
    No-op when PORTFOLIO_API_KEY env var is not set (local dev).
    """
    if PORTFOLIO_API_KEY and x_api_key != PORTFOLIO_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# ─── Market signals cache ─────────────────────────────────────────────────────
# Recompute only when the CSV file changes (mtime-based invalidation).
# Avoids re-running load_nifty_csv + calculate_indicators (~2 s) on every request.
_market_signals_cache: dict = {"df": None, "csv_mtime": -1.0}

# ─── Request Logging Middleware ───────────────────────────────────────────────

_LOG_MAX_ROWS  = 10_000   # rows kept after trimming
_LOG_TRIM_AT   = 15_000   # trim triggered when file exceeds this many rows


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request to api_request_log.csv.
    Columns: timestamp, method, path, status_code, duration_ms

    Rotation: when the file exceeds _LOG_TRIM_AT rows the oldest rows are
    dropped, keeping the most recent _LOG_MAX_ROWS rows.  Checked every
    500 requests so the overhead is negligible.
    """

    _HEADER = ["timestamp", "method", "path", "status_code", "duration_ms"]

    def __init__(self, app, log_path: str = API_LOG_PATH):
        super().__init__(app)
        self.log_path = log_path
        self._req_count = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._HEADER)

    def _rotate_if_needed(self) -> None:
        """Trim the log file to _LOG_MAX_ROWS when it grows too large."""
        try:
            with open(self.log_path, "r", newline="", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= _LOG_TRIM_AT + 1:   # +1 for header
                return
            kept = [lines[0]] + lines[-_LOG_MAX_ROWS:]
            tmp = self.log_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(kept)
            os.replace(tmp, self.log_path)
        except Exception:
            pass

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                ])
            self._req_count += 1
            if self._req_count % 500 == 0:
                self._rotate_if_needed()
        except Exception:
            pass
        return response


# ─── Singleton orchestrator ───────────────────────────────────────────────────

_orchestrator: Optional[WorkflowOrchestrator] = None


def get_orchestrator() -> WorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        active_csv = _get_active_csv()
        if not os.path.exists(active_csv):
            raise RuntimeError(f"Data CSV not found: {active_csv}. Set NIFTY_CSV_PATH env var.")
        _orchestrator = WorkflowOrchestrator(active_csv, OUTPUT_DIR)
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up orchestrator on startup (loads CSV + computes indicators once)
    get_orchestrator()
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RITA — Nifty 50 RL Investment API",
    description=(
        "Double DQN investment system for Nifty 50 index. "
        "8-step workflow: goal → market → strategy → train → period → backtest → results → update."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware, log_path=API_LOG_PATH)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class GoalRequest(BaseModel):
    target_return_pct: float = Field(15.0, description="Target annual return %", ge=1.0, le=100.0)
    time_horizon_days: int = Field(365, description="Investment horizon in days", ge=30)
    risk_tolerance: str = Field("moderate", description="conservative | moderate | aggressive")


class TrainRequest(BaseModel):
    timesteps: int = Field(200_000, description="Training timesteps", ge=10_000)
    force_retrain: bool = Field(False, description="Force retrain even if model exists")
    model_type: str = Field("bull", description="Model type: bull | bear | both")


class PeriodRequest(BaseModel):
    start: str = Field(BACKTEST_START, description="Simulation start date (YYYY-MM-DD)")
    end: Optional[str] = Field(None, description="Simulation end date (YYYY-MM-DD), null = latest")


class StepResponse(BaseModel):
    step: int
    name: str
    status: str = "ok"
    result: dict


def _wrap(step_result: dict) -> dict:
    """Ensure result is JSON-serialisable (convert non-serialisable types to str)."""
    import json
    return json.loads(json.dumps(step_result, default=str))


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None for JSON safety."""
    import math
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """
    Rich service health check.

    Returns model age, data freshness, last pipeline run, and Sharpe trend
    in addition to the basic liveness indicators.
    """
    from datetime import datetime

    orch = get_orchestrator()
    model_path = os.path.join(OUTPUT_DIR, "rita_ddqn_model.zip")
    model_exists = os.path.exists(model_path)

    # Model age
    model_age_days: Optional[float] = None
    if model_exists:
        mtime = os.path.getmtime(model_path)
        model_age_days = round((time.time() - mtime) / 86400, 1)

    # Last pipeline run (latest timestamp in monitor_log.csv)
    last_run: Optional[str] = None
    monitor_path = os.path.join(OUTPUT_DIR, "monitor_log.csv")
    if os.path.exists(monitor_path):
        try:
            import pandas as pd
            mdf = pd.read_csv(monitor_path)
            mdf.columns = [c.strip() for c in mdf.columns]
            ended = mdf[mdf["status"] == "completed"]
            if not ended.empty and "ended_at" in ended.columns:
                last_run = ended["ended_at"].dropna().iloc[-1]
        except Exception:
            pass

    # Sharpe trend (last 5 backtest rounds)
    sharpe_trend: list = []
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    if os.path.exists(history_path):
        try:
            import pandas as pd
            hdf = pd.read_csv(history_path)
            if "backtest_sharpe" in hdf.columns:
                sharpe_trend = hdf["backtest_sharpe"].tail(5).dropna().tolist()
        except Exception:
            pass

    # Data freshness
    detector = DriftDetector(OUTPUT_DIR, CSV_PATH)
    freshness = detector.check_data_freshness()

    return {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv_loaded": orch._raw_df is not None,
        "model_exists": model_exists,
        "model_age_days": model_age_days,
        "output_dir": OUTPUT_DIR,
        "last_pipeline_run": last_run,
        "sharpe_trend_last5": sharpe_trend,
        "data_freshness": freshness,
    }


@app.get("/progress", tags=["System"])
def progress():
    """Return pipeline progress summary."""
    return get_orchestrator().session.get_progress_summary()


@app.post("/reset", tags=["System"])
def reset():
    """Reset orchestrator session (clears in-memory state, keeps saved files)."""
    global _orchestrator
    _orchestrator = WorkflowOrchestrator(_get_active_csv(), OUTPUT_DIR)
    return {"status": "reset", "message": "Orchestrator session cleared."}


# ─── Metrics ──────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["System"])
def metrics():
    """
    Aggregated performance + pipeline metrics in JSON format.

    Includes: latest backtest KPIs, training history summary, step timing
    averages, and API request log statistics.
    """
    import pandas as pd

    result: dict = {
        "pipeline": {},
        "training": {},
        "api_requests": {},
    }

    # Pipeline step timing
    monitor_path = os.path.join(OUTPUT_DIR, "monitor_log.csv")
    if os.path.exists(monitor_path):
        try:
            mdf = pd.read_csv(monitor_path)
            mdf.columns = [c.strip() for c in mdf.columns]
            completed = mdf[mdf["status"] == "completed"]
            if "step_num" in completed.columns and "duration_secs" in completed.columns:
                completed = completed.copy()
                completed["duration_secs"] = pd.to_numeric(completed["duration_secs"], errors="coerce")
                timing = (
                    completed.groupby("step_num")["duration_secs"]
                    .agg(["mean", "max", "count"])
                    .round(2)
                    .to_dict(orient="index")
                )
                result["pipeline"] = {
                    "total_logged_steps": len(mdf),
                    "completed_steps": len(completed),
                    "failed_steps": int((mdf["status"] == "failed").sum()),
                    "step_timing": timing,
                }
        except Exception as exc:
            result["pipeline"]["error"] = str(exc)

    # Training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    if os.path.exists(history_path):
        try:
            hdf = pd.read_csv(history_path)
            if not hdf.empty:
                latest = hdf.iloc[-1]
                result["training"] = {
                    "rounds": len(hdf),
                    "latest_backtest_sharpe": round(float(latest.get("backtest_sharpe", 0)), 4),
                    "latest_backtest_mdd_pct": round(float(latest.get("backtest_mdd_pct", 0)), 2),
                    "latest_backtest_cagr_pct": round(float(latest.get("backtest_cagr_pct", 0)), 2),
                    "latest_constraints_met": bool(latest.get("backtest_constraints_met", False)),
                    "avg_backtest_sharpe": round(float(hdf["backtest_sharpe"].mean()), 4),
                }
        except Exception as exc:
            result["training"]["error"] = str(exc)

    # API request stats
    detector = DriftDetector(OUTPUT_DIR, CSV_PATH)
    result["api_requests"] = detector.api_request_stats()

    return _sanitize(result)


# ─── Drift ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/drift", tags=["Observability"])
def check_drift():
    """
    Model drift and health report.

    Runs all DriftDetector checks:
    - Sharpe drift (vs rolling mean)
    - Return degradation (consecutive CAGR decline)
    - Data freshness (days since latest CSV date)
    - Pipeline health (step failure rate)
    - Constraint breach rate

    Returns a full report + overall health badge (ok / warn / alert).
    """
    try:
        detector = DriftDetector(OUTPUT_DIR, CSV_PATH)
        report   = detector.full_report()
        summary  = detector.health_summary(report)
        return _sanitize({"health": summary, "report": report})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 1 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/goal", response_model=StepResponse, tags=["Workflow"])
def set_goal(body: GoalRequest):
    """
    **Step 1** — Set financial goal.

    Validates the target return against Nifty 50 historical performance and returns
    a feasibility assessment.
    """
    try:
        result = get_orchestrator().step1_set_goal(
            body.target_return_pct, body.time_horizon_days, body.risk_tolerance
        )
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 2 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/market", response_model=StepResponse, tags=["Workflow"])
def analyze_market():
    """
    **Step 2** — Analyze current market conditions.

    Computes RSI, MACD, Bollinger Bands, ATR, EMA on the latest 252 trading days.
    Returns trend classification and sentiment proxy.
    """
    try:
        result = get_orchestrator().step2_analyze_market()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 3 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/strategy", response_model=StepResponse, tags=["Workflow"])
def design_strategy():
    """
    **Step 3** — Design investment strategy.

    Combines market research with the financial goal to select an allocation approach.
    Requires step1 and step2 to be completed first.
    """
    try:
        result = get_orchestrator().step3_design_strategy()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 4 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/train", response_model=StepResponse, tags=["Workflow"])
def train_model(body: TrainRequest):
    """
    **Step 4** — Train the Double DQN model.

    Trains on 2010–2022 data, validates on 2023–2024.
    By default reuses an existing trained model if one is saved — set `force_retrain=true`
    to retrain from scratch (~6 min on CPU).
    """
    try:
        result = get_orchestrator().step4_train_model(
            timesteps=body.timesteps, force_retrain=body.force_retrain, model_type=body.model_type
        )
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 5 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/period", response_model=StepResponse, tags=["Workflow"])
def set_period(body: PeriodRequest):
    """
    **Step 5** — Set the backtest simulation period.

    Default: 2025-01-01 → latest available date in the CSV.
    Must not overlap the training period (2010–2022).
    """
    try:
        result = get_orchestrator().step5_set_simulation_period(body.start, body.end)
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 6 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/backtest", response_model=StepResponse, tags=["Workflow"])
def run_backtest():
    """
    **Step 6** — Run the backtest.

    Runs the trained DDQN model through the simulation period day by day.
    Returns performance metrics vs Nifty Buy-and-Hold benchmark.
    Requires steps 4 and 5 to be completed first.
    """
    try:
        result = get_orchestrator().step6_run_backtest()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 7 ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/results", response_model=StepResponse, tags=["Workflow"])
def get_results():
    """
    **Step 7** — Get full results report.

    Generates all 5 performance and interpretability plots, checks constraints
    (Sharpe > 1.0, MDD < 10%), and returns the complete performance summary.
    """
    try:
        result = get_orchestrator().step7_get_results()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 8 ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/goal/update", response_model=StepResponse, tags=["Workflow"])
def update_goal():
    """
    **Step 8** — Update financial goal based on backtest results.

    Compares actual return vs target and suggests a revised goal for the next cycle.
    Closes the learning loop.
    """
    try:
        result = get_orchestrator().step8_update_goal()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Full pipeline ────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    target_return_pct: float = Field(15.0, ge=1.0, le=100.0)
    time_horizon_days: int = Field(365, ge=30)
    risk_tolerance: str = Field("moderate")
    timesteps: int = Field(200_000, ge=10_000)
    force_retrain: bool = Field(False)
    sim_start: str = Field(BACKTEST_START)
    sim_end: Optional[str] = Field(None)


@app.post("/api/v1/pipeline", tags=["Workflow"])
def run_pipeline(body: PipelineRequest):
    """
    **Full pipeline** — Run all 8 steps sequentially.

    Convenience endpoint for automated runs. Returns results from all steps.
    """
    try:
        orch = get_orchestrator()
        config = {
            "target_return_pct": body.target_return_pct,
            "time_horizon_days": body.time_horizon_days,
            "risk_tolerance": body.risk_tolerance,
            "timesteps": body.timesteps,
            "force_retrain": body.force_retrain,
            "sim_start": body.sim_start,
            "sim_end": body.sim_end,
        }
        # Run steps individually so force_retrain is honoured
        results = {}
        results["step1"] = orch.step1_set_goal(
            config["target_return_pct"], config["time_horizon_days"], config["risk_tolerance"]
        )
        results["step2"] = orch.step2_analyze_market()
        results["step3"] = orch.step3_design_strategy()
        results["step4"] = orch.step4_train_model(
            timesteps=config["timesteps"], force_retrain=config["force_retrain"]
        )
        results["step5"] = orch.step5_set_simulation_period(config["sim_start"], config["sim_end"])
        results["step6"] = orch.step6_run_backtest()
        results["step7"] = orch.step7_get_results()
        results["step8"] = orch.step8_update_goal()
        results["progress"] = orch.session.get_progress_summary()
        return _wrap(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Data endpoints for HTML dashboard ────────────────────────────────────────

def _df_to_json(df):
    """Convert DataFrame to JSON-safe list of dicts (NaN → null via pandas encoder)."""
    import json
    return json.loads(df.to_json(orient="records"))


# ─── Data Preparation ─────────────────────────────────────────────────────────

@app.get("/api/v1/data-prep/status", tags=["Data"])
def data_prep_status():
    """
    Return current data status: input files available, active CSV info,
    and whether an extended (prepared) CSV already exists.
    """
    import glob as _glob
    import pandas as pd

    # Input files
    input_files = sorted(_glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    file_info = []
    for f in input_files:
        stat = os.stat(f)
        file_info.append({
            "name": os.path.basename(f),
            "size_kb": round(stat.st_size / 1024, 1),
        })

    # Active CSV info
    active_csv = _get_active_csv()
    csv_info = {"path": active_csv, "source": "prepared" if os.path.exists(_PREPARED_CSV) else "base"}
    if os.path.exists(active_csv):
        try:
            df = pd.read_csv(active_csv, usecols=[0])
            dates = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
            csv_info.update({
                "total_rows": len(dates),
                "date_from": str(dates.min())[:10],
                "date_to":   str(dates.max())[:10],
            })
        except Exception as exc:
            csv_info["error"] = str(exc)

    return {
        "input_dir": INPUT_DIR,
        "input_files": file_info,
        "active_csv": csv_info,
        "prepared_csv_exists": os.path.exists(_PREPARED_CSV),
    }


@app.post("/api/v1/data-prep/run", tags=["Data"])
def data_prep_run():
    """
    Merge all CSV files in rita_input/ with the base merged CSV and save
    the result to rita_output/nifty_merged.csv.

    After a successful run the orchestrator is reset so subsequent pipeline
    steps automatically use the extended dataset.
    """
    from rita.core.data_loader import prepare_data

    result = prepare_data(
        input_dir  = INPUT_DIR,
        base_csv   = CSV_PATH,
        output_csv = _PREPARED_CSV,
    )

    if result.get("status") == "ok":
        # Reset orchestrator + market signals cache so they pick up new data
        global _orchestrator
        _orchestrator = None
        _market_signals_cache["df"] = None
        _market_signals_cache["csv_mtime"] = -1.0

    return result


@app.get("/api/v1/backtest-daily", tags=["Data"])
def get_backtest_daily():
    """Serve backtest_daily.csv as JSON for the HTML dashboard."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "backtest_daily.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="backtest_daily.csv not found — run pipeline first")
    return _df_to_json(pd.read_csv(path))


@app.get("/api/v1/performance-summary", tags=["Data"])
def get_performance_summary():
    """Serve performance_summary.csv as a flat JSON dict."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "performance_summary.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="performance_summary.csv not found — run pipeline first")
    df = pd.read_csv(path)
    return df.set_index("metric")["value"].to_dict()


@app.get("/api/v1/training-history", tags=["Data"])
def get_training_history():
    """Serve training_history.csv as JSON."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "training_history.csv")
    if not os.path.exists(path):
        return []
    return _df_to_json(pd.read_csv(path))


@app.get("/api/v1/mcp-calls", tags=["Data"])
def get_mcp_calls(limit: int = 100):
    """Serve mcp_call_log.csv as JSON (latest N calls, newest first)."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "mcp_call_log.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    tail = df.tail(limit).iloc[::-1].reset_index(drop=True)
    return _df_to_json(tail)


@app.get("/api/v1/step-log", tags=["Data"])
def get_step_log():
    """Serve monitor_log.csv as JSON."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "monitor_log.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return _df_to_json(df)


@app.get("/api/v1/shap", tags=["Data"])
def get_shap():
    """Serve shap_importance.csv as JSON (feature, importance columns)."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "shap_importance.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "feature"}, inplace=True)
    return _df_to_json(df)


@app.get("/api/v1/risk-timeline", tags=["Data"])
def get_risk_timeline():
    """Serve risk_timeline.csv as JSON (backtest phase only)."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "risk_timeline.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if "phase" in df.columns:
        df = df[df["phase"] == "Backtest"]
    return _df_to_json(df)


@app.get("/api/v1/market-signals", tags=["Data"])
def get_market_signals(timeframe: str = "daily", periods: int = 300):
    """
    Technical indicator time series for the HTML dashboard Market Signals page.

    timeframe: daily | weekly | monthly
    periods:   number of bars to return (default 300)

    The daily indicator DataFrame is cached in memory and only recomputed when
    the source CSV changes on disk (mtime-based invalidation).
    """
    try:
        from rita.core.data_loader import load_nifty_csv
        from rita.core.technical_analyzer import calculate_indicators

        # --- Cache check (daily base only; weekly/monthly derived from it) ---
        active_csv = _get_active_csv()
        csv_mtime = os.path.getmtime(active_csv) if os.path.exists(active_csv) else -1.0
        if _market_signals_cache["df"] is None or _market_signals_cache["csv_mtime"] != csv_mtime:
            raw = load_nifty_csv(active_csv)
            _market_signals_cache["df"] = calculate_indicators(raw)
            _market_signals_cache["csv_mtime"] = csv_mtime

        df = _market_signals_cache["df"]

        if timeframe == "weekly":
            df = df.resample("W").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna(subset=["Close"])
            df = calculate_indicators(df)
        elif timeframe == "monthly":
            df = df.resample("ME").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna(subset=["Close"])
            df = calculate_indicators(df)

        df = df.tail(periods).copy()
        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = df["date"].astype(str).str[:10]

        keep = [c for c in [
            "date", "Close", "Volume", "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid", "bb_pct_b",
            "ema_5", "ema_13", "ema_26", "ema_50", "ema_200",
            "atr_14", "trend_score",
        ] if c in df.columns]

        return _df_to_json(df[keep])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Portfolio Manager ────────────────────────────────────────────────────────

@app.get("/api/v1/portfolio/summary", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_portfolio_summary():
    """
    Combined portfolio summary for the FnO Portfolio Manager dashboard.

    Returns positions, greeks, margin, stress scenarios, payoff curve,
    market data and scenario levels — all computed from CSV files in rita_input/.

    **Auth:** set the `PORTFOLIO_API_KEY` env var to require an `X-API-Key`
    header on every request.  If the env var is not set the endpoint is open
    (suitable for local-only use).
    """
    try:
        from rita.core.portfolio_manager import get_portfolio_summary as _build
        return _build(INPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User's natural-language investment question")
    portfolio_inr: float = Field(default=1_000_000, ge=1, description="Portfolio size in INR for stress/comparison scenarios")


@app.post("/api/v1/chat", tags=["Chat"])
def chat(req: ChatRequest):
    """
    Classify a free-text investment query and return a deterministic OHLCV-driven response.

    No Claude API call at runtime.  Uses all-MiniLM-L6-v2 cosine similarity to
    route to one of 20 fixed investment scenarios, then runs the matching core handler.

    Returns intent name, confidence score, and the filled template response.
    """
    import time as _time
    from rita.core.data_loader import load_nifty_csv
    from rita.core.technical_analyzer import calculate_indicators
    from rita.core.classifier import classify, dispatch
    from rita.core.chat_monitor import log_query as _log_chat

    t0 = _time.perf_counter()

    # Reuse the market-signals df cache (loaded + indicators applied once per CSV change)
    try:
        active_csv = _get_active_csv()
        csv_mtime = os.path.getmtime(active_csv) if os.path.exists(active_csv) else -1.0
        if _market_signals_cache["df"] is None or _market_signals_cache["csv_mtime"] != csv_mtime:
            raw = load_nifty_csv(active_csv)
            _market_signals_cache["df"] = calculate_indicators(raw)
            _market_signals_cache["csv_mtime"] = csv_mtime
        df = _market_signals_cache["df"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load market data: {e}")

    try:
        result = classify(req.query)
        response_text = dispatch(result, df, portfolio_inr=req.portfolio_inr, output_dir=OUTPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    latency_ms = (_time.perf_counter() - t0) * 1000
    status = "low_confidence" if result.low_confidence else "success"

    _log_chat(
        query_text=req.query,
        intent_name=result.intent.name,
        handler=result.intent.handler,
        confidence=result.confidence,
        low_confidence=result.low_confidence,
        latency_ms=latency_ms,
        response_preview=response_text[:200],
        status=status,
    )

    return {
        "intent":          result.intent.name,
        "handler":         result.intent.handler,
        "confidence":      round(result.confidence, 4),
        "low_confidence":  result.low_confidence,
        "response":        response_text,
        "latency_ms":      round(latency_ms, 1),
    }


@app.get("/api/v1/chat/monitor", tags=["Chat"])
def chat_monitor_summary():
    """KPIs and recent queries from the chat monitor CSV."""
    from rita.core.chat_monitor import get_summary, get_recent_queries, get_intent_distribution
    return {
        "summary": get_summary(),
        "recent":  get_recent_queries(20),
        "intents": get_intent_distribution(),
    }


# ─── Static dashboard files ────────────────────────────────────────────────────

_DASHBOARD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "dashboard")
)
if os.path.isdir(_DASHBOARD_DIR):
    from fastapi.staticfiles import StaticFiles as _SF
    app.mount("/dashboard", _SF(directory=_DASHBOARD_DIR, html=True), name="dashboard")
