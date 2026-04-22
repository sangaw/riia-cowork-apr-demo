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
from datetime import datetime

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
    TRAIN_TIMESTEPS,
)

# ─── Configuration ────────────────────────────────────────────────────────────

API_LOG_PATH = os.path.join(OUTPUT_DIR, "api_request_log.csv")

# Path where prepare_data() writes the extended merged CSV (legacy default)
_PREPARED_CSV = os.path.join(OUTPUT_DIR, "nifty_merged.csv")

# ─── Active instrument state ──────────────────────────────────────────────────

_active_instrument_id: str = "nifty50"   # default on startup


def _get_instrument_registry() -> list:
    """Load instruments.json from rita_input/. Returns [] if missing."""
    import json as _json
    registry_path = os.path.join(INPUT_DIR, "instruments.json")
    if not os.path.exists(registry_path):
        return []
    with open(registry_path, "r", encoding="utf-8") as f:
        return _json.load(f)


def _get_active_instrument() -> dict | None:
    """Return the registry entry for the currently selected instrument."""
    for inst in _get_instrument_registry():
        if inst["id"] == _active_instrument_id:
            return inst
    return None


def _get_active_csv() -> str:
    """
    Return the prepared CSV for the active instrument.
    Falls back to the legacy nifty_merged.csv / CSV_PATH if registry is missing.
    """
    inst = _get_active_instrument()
    if inst:
        csv_path = os.path.join(OUTPUT_DIR, inst["prepared_csv"])
        if os.path.exists(csv_path):
            return csv_path
    # Legacy fallback
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

# ─── Run context helpers ──────────────────────────────────────────────────────
_RUN_CONTEXT_PATH = os.path.join(OUTPUT_DIR, "run_context.json")


def _save_run_context() -> None:
    """Persist which instrument the current output files (backtest, perf) belong to."""
    import json as _json
    with open(_RUN_CONTEXT_PATH, "w", encoding="utf-8") as f:
        _json.dump({"instrument_id": _active_instrument_id}, f)


def _get_run_instrument_id() -> str:
    """Return the instrument_id the output files were last written for."""
    import json as _json
    try:
        with open(_RUN_CONTEXT_PATH, "r", encoding="utf-8") as f:
            return _json.load(f).get("instrument_id", "nifty50")
    except (FileNotFoundError, ValueError):
        return "nifty50"

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
        inst = _get_active_instrument()
        rfr = inst.get("risk_free_rate", 0.07) if inst else 0.07
        _orchestrator = WorkflowOrchestrator(
            active_csv, OUTPUT_DIR, risk_free_rate=rfr, instrument_id=_active_instrument_id
        )
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
    timesteps: int = Field(TRAIN_TIMESTEPS, description="Training timesteps", ge=10_000)
    force_retrain: bool = Field(False, description="Force retrain even if model exists")


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
    model_path = orch.model_zip
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
    detector = DriftDetector(OUTPUT_DIR, _get_active_csv())
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
    inst = _get_active_instrument()
    rfr = inst.get("risk_free_rate", 0.07) if inst else 0.07
    _orchestrator = WorkflowOrchestrator(
        _get_active_csv(), OUTPUT_DIR, risk_free_rate=rfr, instrument_id=_active_instrument_id
    )
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
    detector = DriftDetector(OUTPUT_DIR, _get_active_csv())
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
        detector = DriftDetector(OUTPUT_DIR, _get_active_csv())
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
            timesteps=body.timesteps, force_retrain=body.force_retrain
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
        _save_run_context()
        return _wrap(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Step 7 ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/results", response_model=StepResponse, tags=["Workflow"])
def get_results(record_history: bool = True):
    """
    **Step 7** — Get full results report.

    Generates all 5 performance and interpretability plots, checks constraints
    (Sharpe > 1.0, MDD < 10%), and returns the complete performance summary.

    Pass `record_history=false` for scenario/exploration runs to avoid
    polluting the training history log.
    """
    try:
        result = get_orchestrator().step7_get_results(record_to_history=record_history)
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
    timesteps: int = Field(TRAIN_TIMESTEPS, ge=10_000)
    seeds: int = Field(3, ge=1, le=20)
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
            "seeds": body.seeds,
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
            timesteps=config["timesteps"], force_retrain=config["force_retrain"],
            n_seeds=config.get("seeds", 3)
        )
        results["step5"] = orch.step5_set_simulation_period(config["sim_start"], config["sim_end"])
        results["step6"] = orch.step6_run_backtest()
        _save_run_context()
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

@app.get("/api/v1/instruments", tags=["Data"])
def list_instruments():
    """
    Return the instrument registry with data availability status.
    data_ready=true means the prepared CSV exists in rita_output/.
    """
    import json as _json
    registry_path = os.path.join(INPUT_DIR, "instruments.json")
    if not os.path.exists(registry_path):
        return []
    with open(registry_path, "r", encoding="utf-8") as f:
        instruments = _json.load(f)
    for inst in instruments:
        csv_path = os.path.join(OUTPUT_DIR, inst["prepared_csv"])
        inst["data_ready"] = os.path.exists(csv_path)
        inst["csv_path"] = csv_path if inst["data_ready"] else None
    return instruments


@app.get("/api/v1/instrument/active", tags=["Data"])
def get_active_instrument():
    """Return the currently selected instrument with its data_ready status."""
    inst = _get_active_instrument()
    if not inst:
        return {"id": "nifty50", "name": "Nifty 50", "data_ready": True}
    csv_path = os.path.join(OUTPUT_DIR, inst["prepared_csv"])
    inst = dict(inst)
    inst["data_ready"] = os.path.exists(csv_path)
    return inst


@app.post("/api/v1/instrument/select", tags=["Data"])
def select_instrument(body: dict):
    """
    Set the active instrument. Resets the orchestrator and market signals cache
    so subsequent calls load the new instrument's data.

    Body: { "instrument_id": "nifty50" }
    """
    global _active_instrument_id, _orchestrator
    instrument_id = body.get("instrument_id", "").strip()
    if not instrument_id:
        raise HTTPException(status_code=400, detail="instrument_id is required")

    registry = _get_instrument_registry()
    match = next((i for i in registry if i["id"] == instrument_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Unknown instrument: {instrument_id}")

    csv_path = os.path.join(OUTPUT_DIR, match["prepared_csv"])
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=422,
            detail=f"No data for {match['name']} — prepare the CSV first ({match['prepared_csv']})"
        )

    _active_instrument_id = instrument_id
    _orchestrator = None  # force re-init with new CSV on next request
    _market_signals_cache["df"] = None
    _market_signals_cache["csv_mtime"] = -1.0

    return {"status": "ok", "active_instrument": match["id"], "name": match["name"]}


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
    """Serve performance_summary.csv as a flat JSON dict, annotated with instrument context."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "performance_summary.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="performance_summary.csv not found — run pipeline first")
    df = pd.read_csv(path)
    result = df.set_index("metric")["value"].to_dict()
    result["_run_instrument_id"] = _get_run_instrument_id()
    result["_active_instrument_id"] = _active_instrument_id
    return result


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
    # Keep only the last row per step_num (start_step writes in_progress, end_step writes completed/failed)
    if "step_num" in df.columns:
        df = df.drop_duplicates(subset=["step_num"], keep="last")
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


@app.get("/api/v1/data-understanding", tags=["Data"])
def get_data_understanding(instrument_id: str = ""):
    """
    Compute and return data understanding statistics and chart data for the
    selected instrument's prepared CSV.

    Returns:
      - summary: row count, feature count, date range, missing %
      - distributions: histogram bins/values for 8 key features
      - correlation: feature names + correlation matrix
      - timeseries: sampled dates + close, volume, rsi, macd series
      - clustering: elbow inertia (k=2..8) + PCA 2D scatter with cluster labels
    """
    import pandas as pd
    import numpy as np

    # ── Resolve CSV ──────────────────────────────────────────────────────────
    if instrument_id:
        registry = _get_instrument_registry()
        match = next((i for i in registry if i["id"] == instrument_id), None)
        if match:
            candidate = os.path.join(OUTPUT_DIR, match["prepared_csv"])
            csv_path = candidate if os.path.exists(candidate) else _get_active_csv()
        else:
            csv_path = _get_active_csv()
    else:
        csv_path = _get_active_csv()

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No prepared CSV found. Run Data Prep first.")

    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Rename NSE-style volume column
    if "shares traded" in df.columns and "volume" not in df.columns:
        df.rename(columns={"shares traded": "volume"}, inplace=True)

    # Parse date
    date_col = next((c for c in df.columns if c in ("date", "datetime", "time")), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # ── Compute indicators if not present (e.g. raw Nifty CSV) ───────────────
    if "rsi_14" not in df.columns and "close" in df.columns:
        try:
            from rita.core.technical_analyzer import calculate_indicators
            # calculate_indicators expects title-case OHLCV columns
            ohlcv_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            df_titled = df.rename(columns={k: v for k, v in ohlcv_map.items() if k in df.columns})
            df_ind = calculate_indicators(df_titled)
            df_ind.columns = [c.lower() for c in df_ind.columns]
            df = df_ind
        except Exception:
            pass  # proceed with what we have

    # Add log_return if not present
    if "log_return" not in df.columns and "close" in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # ── Summary ──────────────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    total_cells = numeric_df.size
    missing_pct = round(numeric_df.isna().sum().sum() / total_cells * 100, 2) if total_cells else 0.0

    trend_classes = 0
    for col in ("trend_code", "final_trend", "dow_trend_spec"):
        if col in df.columns:
            trend_classes = int(df[col].nunique())
            break

    summary = {
        "rows": len(df),
        "features": len(numeric_df.columns),
        "date_from": str(df[date_col].min())[:10] if date_col else "",
        "date_to":   str(df[date_col].max())[:10] if date_col else "",
        "missing_pct": missing_pct,
        "trend_classes": trend_classes,
    }

    # ── Distributions (histogram) ────────────────────────────────────────────
    # Column aliases: prefer first name found in df
    def _first(candidates):
        return next((c for c in candidates if c in df.columns), None)

    DIST_FEATURES = [
        (_first(["close"]),                          "Close Price"),
        (_first(["daily_return"]),                   "Daily Return"),
        (_first(["rsi_14"]),                         "RSI (14)"),
        (_first(["volume"]),                         "Volume"),
        (_first(["macd", "macd_12_26"]),             "MACD"),
        (_first(["log_return"]),                     "Log Return"),
        (_first(["macd_hist", "macd_histogram_12_26"]), "MACD Histogram"),
        (_first(["bb_pct_b", "volatility_20"]),      "BB %B / Volatility"),
    ]
    distributions = {}
    for col, label in DIST_FEATURES:
        if not col:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        counts, edges = np.histogram(series, bins=40)
        centers = [round(float((edges[i] + edges[i + 1]) / 2), 4) for i in range(len(edges) - 1)]
        distributions[col] = {
            "label": label,
            "labels": centers,
            "values": [int(v) for v in counts],
        }

    # ── Correlation ──────────────────────────────────────────────────────────
    CORR_CANDIDATES = [
        "close", "daily_return", "log_return", "rsi_14",
        "macd", "macd_12_26", "macd_hist", "macd_histogram_12_26",
        "bb_pct_b", "volatility_20", "volume", "trend_code",
        "ema_ratio", "trend_score", "atr_14",
    ]
    CORR_FEATURES = [c for c in CORR_CANDIDATES if c in df.columns]
    corr_matrix = []
    if len(CORR_FEATURES) >= 2:
        corr = df[CORR_FEATURES].corr().round(2)
        corr_matrix = corr.values.tolist()

    correlation = {
        "features": CORR_FEATURES,
        "matrix": corr_matrix,
    }

    # ── Time series (sampled to ≤500 points) ─────────────────────────────────
    MAX_TS = 500
    ts_df = df.copy()
    if date_col:
        ts_df = ts_df.dropna(subset=[date_col])
    if len(ts_df) > MAX_TS:
        step = len(ts_df) // MAX_TS
        ts_df = ts_df.iloc[::step].reset_index(drop=True)

    def _ts_series(col):
        if col not in ts_df.columns:
            return []
        return [None if pd.isna(v) else round(float(v), 4) for v in ts_df[col]]

    macd_col = _first(["macd", "macd_12_26"])
    timeseries = {
        "dates":  [str(v)[:10] for v in ts_df[date_col]] if date_col else [],
        "close":  _ts_series("close"),
        "volume": _ts_series("volume"),
        "rsi":    _ts_series("rsi_14"),
        "macd":   _ts_series(macd_col) if macd_col else [],
    }

    # ── Clustering ───────────────────────────────────────────────────────────
    clustering = {"elbow": {}, "pca": {}}
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        CLUSTER_FEATURES = [
            c for c in [
                "daily_return", "volume", "log_return",
                "ema_5", "ema_50", "atr_14", "bb_pct_b",
                "rsi_14", "macd", "macd_12_26", "ema_ratio", "trend_score",
            ] if c in df.columns
        ]
        if len(CLUSTER_FEATURES) >= 3:
            X = df[CLUSTER_FEATURES].dropna()
            if len(X) >= 20:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Elbow
                inertia = []
                k_range = list(range(2, 9))
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    km.fit(X_scaled)
                    inertia.append(round(float(km.inertia_), 2))
                clustering["elbow"] = {"k": k_range, "inertia": inertia}

                # PCA scatter (k=3)
                km3 = KMeans(n_clusters=3, random_state=42, n_init=5)
                labels = km3.fit_predict(X_scaled)
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_scaled)
                # Sample down for payload size
                idx = np.linspace(0, len(coords) - 1, min(300, len(coords)), dtype=int)
                clustering["pca"] = {
                    "x":       [round(float(coords[i, 0]), 3) for i in idx],
                    "y":       [round(float(coords[i, 1]), 3) for i in idx],
                    "cluster": [int(labels[i]) for i in idx],
                }
    except ImportError:
        clustering["error"] = "scikit-learn not installed"
    except Exception as exc:
        clustering["error"] = str(exc)

    return {
        "instrument_id": instrument_id or _active_instrument_id,
        "summary": summary,
        "distributions": distributions,
        "correlation": correlation,
        "timeseries": timeseries,
        "clustering": clustering,
    }


@app.get("/api/v1/risk-timeline", tags=["Data"])
def get_risk_timeline(phase: str = "Backtest"):
    """Serve risk_timeline.csv as JSON. Use phase='all' to return all phases."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "risk_timeline.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if "phase" in df.columns and phase.lower() != "all":
        df = df[df["phase"] == phase]
    return _df_to_json(df)


@app.get("/api/v1/trade-events", tags=["Data"])
def get_trade_events():
    """Serve risk_trade_events.csv as JSON."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "risk_trade_events.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return _df_to_json(df)


@app.get("/api/v1/training-progress", tags=["Data"])
def get_training_progress():
    """Serve training_progress.csv (TD loss + reward over timesteps) as JSON."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "training_progress.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).dropna(subset=["timestep"])
    return _df_to_json(df)


# ─── Model Changelog ──────────────────────────────────────────────────────────

_CHANGELOG_COLS = ["date", "version", "category", "change", "notes"]
_CHANGELOG_CATEGORIES = ["Reward", "Hyperparameter", "Feature", "Architecture", "Data", "Other"]
_CHANGELOG_SEED = [
    {"date": "2026-03-17", "version": "v1.0", "category": "Hyperparameter",
     "change": "Baseline: Double DQN, 200k timesteps, lr=1e-4, buffer=50k, net=[128,128], explore=10%",
     "notes": "Original model. Backtest Sharpe 1.191, MDD -4.55%, Return 13.85% vs B&H 16.65%"},
    {"date": "2026-03-17", "version": "v1.1", "category": "Reward",
     "change": "Replace binary cliff reward with Markowitz-style: ret - 50*ret² + smooth DD penalty",
     "notes": "Old reward had 2000x scale mismatch. New reward penalises variance proportionally."},
    {"date": "2026-03-17", "version": "v1.1", "category": "Feature",
     "change": "Add ATR-14 as 8th observation feature (atr_14 / atr_mean, clipped 0-3)",
     "notes": "Gives agent live volatility regime signal. Backward-compat via obs-space shape detection."},
]


@app.get("/api/v1/changelog", tags=["Data"])
def get_changelog():
    """Serve model_changelog.csv as JSON, seeding with defaults if absent."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "model_changelog.csv")
    if not os.path.exists(path):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(_CHANGELOG_SEED, columns=_CHANGELOG_COLS).to_csv(path, index=False)
    df = pd.read_csv(path)
    for col in _CHANGELOG_COLS:
        if col not in df.columns:
            df[col] = ""
    return _df_to_json(df[_CHANGELOG_COLS].iloc[::-1].reset_index(drop=True))


class ChangelogEntry(BaseModel):
    date: str
    version: str = ""
    category: str = "Other"
    change: str
    notes: str = ""


@app.post("/api/v1/changelog", tags=["Data"], status_code=201)
def add_changelog_entry(entry: ChangelogEntry):
    """Append a new entry to model_changelog.csv."""
    import pandas as pd
    path = os.path.join(OUTPUT_DIR, "model_changelog.csv")
    if not os.path.exists(path):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.DataFrame(_CHANGELOG_SEED, columns=_CHANGELOG_COLS)
    else:
        df = pd.read_csv(path)
        for col in _CHANGELOG_COLS:
            if col not in df.columns:
                df[col] = ""
    new_row = pd.DataFrame([{
        "date": entry.date, "version": entry.version,
        "category": entry.category, "change": entry.change, "notes": entry.notes,
    }])
    pd.concat([df, new_row], ignore_index=True).to_csv(path, index=False)
    return {"status": "ok", "message": "Entry added"}


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


@app.get("/api/v1/portfolio/price-history", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_price_history():
    """
    All available daily NIFTY + BANKNIFTY close prices from nifty_manual.csv /
    banknifty_manual.csv.  Used by the dashboard to backfill the Daily Progress
    History table without relying solely on browser localStorage.
    """
    try:
        from rita.core.portfolio_manager import load_price_history
        return load_price_history(INPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/orders-charges", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_orders_charges():
    """
    Compute Zerodha F&O broker transaction charges from all orders-*.csv files.
    Returns list of { date, brokerage, stt, exchange, sebi, stamp, gst, total } oldest-first.
    """
    try:
        from rita.core.portfolio_manager import compute_orders_charges
        return compute_orders_charges(INPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/hedge-history", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_hedge_history():
    """
    Multi-day hedge behaviour analysis across all positions-*.csv files.

    Returns per-day premium bucketing (near-ATM / mid-OTM / far-OTM),
    reactive hedging score, anchor positions, and budget utilisation.
    """
    try:
        from rita.core.hedge_analyzer import compute_hedge_history
        return compute_hedge_history(INPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/hedge-calendar", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_hedge_calendar(
    nifty_capital:     float = 250_000,
    banknifty_capital: float = 250_000,
    max_loss_pct:      float = 0.20,
    put_delta:         float = 0.20,
    call_delta:        float = 0.20,
):
    """
    Rolling 3-month collar calendar using Black's 1976 model.

    For each of the next 3 NSE monthly expiries, returns the fair value of:
      • Buy Put  — at the strike where Black's |delta| == put_delta
      • Sell Call — at the strike where Black's delta == call_delta
    and the net collar cost / protected range for NIFTY and BANKNIFTY.

    The 10th-of-month rule applies: once past the 10th, the current month
    is dropped and the window advances to the next 3 forward months.
    """
    try:
        from rita.core.portfolio_manager import compute_hedge_calendar
        return compute_hedge_calendar(
            INPUT_DIR,
            nifty_capital=nifty_capital,
            banknifty_capital=banknifty_capital,
            max_loss_pct=max_loss_pct,
            put_delta=put_delta,
            call_delta=call_delta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import json as _json

# ─── Manoeuvre group state helpers ───────────────────────────────────────────

def _man_groups_path(und: str, month: str) -> str:
    return os.path.join(OUTPUT_DIR, f"man_groups_{und.upper()}_{month.upper()}.json")

def _man_write_history(date, und, month, groups, nifty_spot, banknifty_spot, dte):
    """Shared helper — write/update man_pnl_history.csv rows."""
    hist_path  = os.path.join(OUTPUT_DIR, "man_pnl_history.csv")
    hist_fields = ["date","und","month","group_id","group_name","view",
                   "pnl_now","sl_pnl","target_pnl","lot_count",
                   "nifty_spot","banknifty_spot","dte",
                   "pct_from_sl","pct_from_target"]
    existing: list[dict] = []
    if os.path.exists(hist_path):
        with open(hist_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    new_keys = {(date, und.upper(), month, str(g.get("id","")).strip()) for g in groups}
    kept = [r for r in existing
            if (r["date"], r.get("und","NIFTY"), r["month"], r.get("group_id","")) not in new_keys]
    new_rows = []
    for g in groups:
        new_rows.append({
            "date": date, "und": und.upper(), "month": month,
            "group_id":        str(g.get("id","")).strip(),
            "group_name":      str(g.get("name","")).strip(),
            "view":            str(g.get("view","")).strip(),
            "pnl_now":         g.get("pnl_now",    0),
            "sl_pnl":          g.get("sl_pnl",     ""),
            "target_pnl":      g.get("target_pnl", ""),
            "lot_count":       g.get("lot_count",  0),
            "nifty_spot":      nifty_spot     if nifty_spot     is not None else "",
            "banknifty_spot":  banknifty_spot if banknifty_spot is not None else "",
            "dte":             dte            if dte            is not None else "",
            "pct_from_sl":     g.get("pct_from_sl",    ""),
            "pct_from_target": g.get("pct_from_target",""),
        })
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hist_fields)
        writer.writeheader()
        writer.writerows(kept + new_rows)
    return len(new_rows), len(kept) + len(new_rows)


@app.put("/api/v1/portfolio/man-groups", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def put_man_groups(payload: dict):
    """
    Persist browser group assignments to server so daily cron can snapshot
    without the page being open.

    Payload: ``{"month": "APR", "und": "NIFTY", "groupState": {...}, "assign": {...}}``
    Writes to ``rita_output/man_groups_NIFTY_APR.json``.
    """
    try:
        month = str(payload.get("month","")).strip().upper()
        und   = str(payload.get("und",   "NIFTY")).strip().upper()
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        data = {
            "month":      month,
            "und":        und,
            "groupState": payload.get("groupState", {}),
            "assign":     payload.get("assign",     {}),
            "saved_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(_man_groups_path(und, month), "w", encoding="utf-8") as f:
            _json.dump(data, f)
        return {"status": "ok", "month": month, "und": und}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/man-groups", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_man_groups(month: str = "", und: str = "NIFTY"):
    """
    Load server-side group assignments for an instrument-month combination.
    Returns ``{"month","und","groupState","assign","saved_at"}`` or 404 if not saved yet.
    """
    try:
        month = month.strip().upper()
        und   = und.strip().upper()
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        path = _man_groups_path(und, month)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"No saved groups for {und} {month}")
        with open(path, encoding="utf-8") as f:
            return _json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/man-daily-snapshot", tags=["Portfolio"],
          dependencies=[Depends(_require_portfolio_key)])
def post_man_daily_snapshot(payload: dict):
    """
    Server-triggered daily snapshot — called by scheduled cron at market close.
    Reads group assignments from ``man_groups_{UND}_{MONTH}.json`` and current
    positions from portfolio_manager, then writes man_pnl_history.csv and
    man_session_notes.csv (marks entry as 'auto').

    Payload: ``{"month": "APR", "und": "NIFTY"}``
    If ``und`` is omitted, iterates over both NIFTY and BANKNIFTY.
    """
    try:
        from rita.core.portfolio_manager import get_portfolio_summary as _build
        from datetime import date as _date
        month = str(payload.get("month","")).strip().upper()
        if not month:
            raise HTTPException(status_code=422, detail="month is required")

        und_requested = str(payload.get("und","")).strip().upper()
        unds_to_run   = [und_requested] if und_requested else ["NIFTY", "BANKNIFTY"]

        # Load current portfolio once
        summary        = _build(INPUT_DIR)
        all_positions  = summary.get("positions", [])
        market         = summary.get("market", {})
        sc_levels      = summary.get("scenario_levels", {})
        nifty_spot     = (market.get("NIFTY")     or {}).get("close")
        banknifty_spot = (market.get("BANKNIFTY") or {}).get("close")
        date_str       = (market.get("NIFTY")     or {}).get("date",
                          datetime.utcnow().strftime("%d-%b-%Y"))
        lot_sizes = {"NIFTY": 65, "BANKNIFTY": 30}

        GROUP_DEFS = [
            {"id":"anchor",      "name":"Monthly Anchor",  "defaultView":"bull"},
            {"id":"directional", "name":"Directional",     "defaultView":"bull"},
            {"id":"futures",     "name":"Futures",         "defaultView":"bull"},
            {"id":"spread",      "name":"Spread",          "defaultView":"bull"},
            {"id":"hedge",       "name":"Hedge",           "defaultView":"bear"},
        ]

        def payoff(lot, price):
            sign = 1 if lot.get("side") == "Long" else -1
            t    = lot.get("type","")
            avg  = lot.get("avg", 0)
            sk   = lot.get("strike_val", 0) or 0
            lsz  = lot.get("lotSz", 1)
            if t == "FUT":   intr = price
            elif t == "CE":  intr = max(0, price - sk)
            else:            intr = max(0, sk - price)
            return sign * lsz * (intr - avg)

        def pct(s, l):
            return round(((s - l) / l) * 10000) / 100 if s and l else None

        total_written = 0
        total_rows    = 0
        skipped       = []

        for und in unds_to_run:
            gpath = _man_groups_path(und, month)
            if not os.path.exists(gpath):
                skipped.append(und)
                continue
            with open(gpath, encoding="utf-8") as f:
                gdata = _json.load(f)
            assign      = gdata.get("assign", {})
            group_state = gdata.get("groupState", {})

            # Filter positions to this und+month
            positions = [p for p in all_positions
                         if p.get("exp") == month and p.get("und","").upper() == und]

            # Expand → lots
            all_lots = []
            for p in positions:
                lsz     = lot_sizes.get(und, 1)
                n       = max(1, round(p.get("qty", 0) / lsz))
                pnl_per = p.get("pnl", 0) / n
                for i in range(1, n + 1):
                    lot = dict(p)
                    lot["lotKey"] = f"{p['instrument']}_L{i}"
                    lot["lotSz"]  = lsz
                    lot["lotPnl"] = pnl_per
                    lot["nLots"]  = n
                    all_lots.append(lot)

            ref_spot = banknifty_spot if und == "BANKNIFTY" else nifty_spot

            groups_payload = []
            for gd in GROUP_DEFS:
                gs     = group_state.get(gd["id"], {})
                name   = gs.get("name", gd["name"])
                view   = gs.get("view", gd["defaultView"])
                g_lots = [l for l in all_lots if assign.get(l["lotKey"]) == gd["id"]]
                tot_now = sum(l["lotPnl"] for l in g_lots)
                tot_sl  = sum(payoff(l, (sc_levels.get(und,{}).get(view,{}) or {}).get("sl", l.get("avg",0)))
                              for l in g_lots if (sc_levels.get(und,{}).get(view,{}) or {}).get("sl") is not None)
                tot_tgt = sum(payoff(l, (sc_levels.get(und,{}).get(view,{}) or {}).get("target", l.get("avg",0)))
                              for l in g_lots if (sc_levels.get(und,{}).get(view,{}) or {}).get("target") is not None)
                sc_g    = (sc_levels.get(und,{}).get(view,{}) or {})
                groups_payload.append({
                    "id": gd["id"], "name": name, "view": view,
                    "pnl_now":         round(tot_now),
                    "sl_pnl":          round(tot_sl)  if g_lots else None,
                    "target_pnl":      round(tot_tgt) if g_lots else None,
                    "lot_count":       len(g_lots),
                    "pct_from_sl":     pct(ref_spot, sc_g.get("sl")),
                    "pct_from_target": pct(ref_spot, sc_g.get("target")),
                })

            rw, tot = _man_write_history(
                date_str, und, month, groups_payload, nifty_spot, banknifty_spot, None)
            total_written += rw
            total_rows     = tot

        if skipped and not total_written:
            return {"status": "skipped",
                    "reason": f"No saved group assignments for {', '.join(skipped)} {month}"}

        # Append auto note (once per month trigger)
        notes_path   = os.path.join(OUTPUT_DIR, "man_session_notes.csv")
        notes_fields = ["ts","date","und","month","nifty_spot","banknifty_spot","dte","notes"]
        write_header = not os.path.exists(notes_path)
        with open(notes_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=notes_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date":          date_str,
                "und":           und_requested or "ALL",
                "month":         month,
                "nifty_spot":    nifty_spot     or "",
                "banknifty_spot":banknifty_spot or "",
                "dte":           "",
                "notes":         "auto-snapshot",
            })

        return {"status": "ok", "month": month, "date": date_str,
                "rows_written": total_written, "total_rows": total_rows,
                "skipped": skipped}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/man-daily-status", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_man_daily_status():
    """
    Ops portal status endpoint — returns today's snapshot state, 7-day history,
    action counts, and recent session notes for the Daily Ops page.
    """
    try:
        from rita.core.portfolio_manager import get_portfolio_summary as _build
        from datetime import date as _date

        summary   = _build(INPUT_DIR)
        positions = summary.get("positions", [])
        market    = summary.get("market", {})
        today_str = (_date.today()).strftime("%d-%b-%Y")

        # Active months from live positions
        active_months = sorted({p.get("exp","") for p in positions if p.get("exp")})

        # Snapshot status per month
        hist_path = os.path.join(OUTPUT_DIR, "man_pnl_history.csv")
        hist_rows: list[dict] = []
        if os.path.exists(hist_path):
            with open(hist_path, newline="", encoding="utf-8") as f:
                hist_rows = list(csv.DictReader(f))

        snapshot_status = {}
        for m in active_months:
            m_rows     = [r for r in hist_rows if r.get("month","").upper() == m.upper()]
            today_rows = [r for r in m_rows if r.get("date","") == today_str]
            last_date  = m_rows[-1]["date"] if m_rows else None
            # Groups saved = at least one instrument has saved assignments for this month
            has_groups = any(
                os.path.exists(_man_groups_path(u, m))
                for u in ("NIFTY", "BANKNIFTY")
            )
            snapshot_status[m] = {
                "snapshotted_today": bool(today_rows),
                "last_date":         last_date,
                "groups_saved":      has_groups,
                "lot_count":         sum(int(r.get("lot_count",0) or 0) for r in today_rows),
            }

        # Action log — today's count + last action
        action_path = os.path.join(OUTPUT_DIR, "man_action_log.csv")
        actions_today = 0
        last_action_ts = None
        if os.path.exists(action_path):
            with open(action_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("date","") == today_str:
                        actions_today += 1
                        last_action_ts = row.get("ts")

        # Recent snapshots (last 7 unique dates across all months)
        seen_dates: set = set()
        recent_snapshots = []
        for r in reversed(hist_rows):
            key = (r.get("date",""), r.get("month",""))
            if key not in seen_dates:
                seen_dates.add(key)
                recent_snapshots.append({
                    "date":      r.get("date"),
                    "month":     r.get("month"),
                    "nifty_spot":r.get("nifty_spot",""),
                    "lot_count": r.get("lot_count",""),
                })
            if len(seen_dates) >= 14:
                break
        recent_snapshots.reverse()

        # Recent session notes (last 5)
        notes_path = os.path.join(OUTPUT_DIR, "man_session_notes.csv")
        recent_notes = []
        if os.path.exists(notes_path):
            with open(notes_path, newline="", encoding="utf-8") as f:
                all_notes = list(csv.DictReader(f))
            recent_notes = [
                {"ts": r.get("ts",""), "date": r.get("date",""),
                 "month": r.get("month",""), "notes": r.get("notes","")}
                for r in all_notes[-5:]
            ]

        return {
            "today":            today_str,
            "active_months":    active_months,
            "snapshot_status":  snapshot_status,
            "actions_today":    actions_today,
            "last_action_ts":   last_action_ts,
            "recent_snapshots": recent_snapshots,
            "recent_notes":     recent_notes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/man-snapshot", tags=["Portfolio"],
          dependencies=[Depends(_require_portfolio_key)])
def post_man_snapshot(payload: dict):
    """
    Save a full manoeuvre snapshot — writes three CSVs in one call:

    1. ``rita_output/man_pnl_history.csv``   — group-level P&L + market context
    2. ``rita_output/man_position_snapshot.csv`` — per-lot detail per group
    3. ``rita_output/man_session_notes.csv`` — free-text session note (append-only)

    Payload::

        {
          "month": "APR", "date": "29-Mar-2026",
          "nifty_spot": 22400.50, "banknifty_spot": 48200.0, "dte": 3,
          "notes": "Rolled anchor after breakout",
          "groups": [{
            "id": "anchor", "name": "Monthly Anchor", "view": "bull",
            "pnl_now": -12400, "sl_pnl": -45000, "target_pnl": 38000,
            "lot_count": 3, "pct_from_sl": 2.14, "pct_from_target": -5.60,
            "lots": [{
              "lot_key": "NIFTY25APR22500CE_L1", "instrument": "NIFTY25APR22500CE",
              "und": "NIFTY", "type": "CE", "side": "Long",
              "lot_sz": 65, "avg": 180.0, "pnl_now": -1200,
              "pnl_sl": -3800, "pnl_target": 6500
            }]
          }]
        }

    Re-saving on the same day overwrites existing rows for the same date+month+group_id
    (history + position CSVs).  Notes are always appended.
    """
    try:
        month          = str(payload.get("month", "")).strip().upper()
        und            = str(payload.get("und",   "NIFTY")).strip().upper()
        date           = str(payload.get("date",  "")).strip()
        groups         = payload.get("groups", [])
        nifty_spot     = payload.get("nifty_spot")
        banknifty_spot = payload.get("banknifty_spot")
        dte            = payload.get("dte")
        notes          = str(payload.get("notes", "")).strip()

        if not month or not date or not groups:
            raise HTTPException(status_code=422, detail="month, date and groups are required")

        # ── CSV 1: man_pnl_history.csv (group-level + market context) ─────────
        rows_written, total_rows = _man_write_history(
            date, und, month, groups, nifty_spot, banknifty_spot, dte)

        # ── CSV 2: man_position_snapshot.csv (per-lot detail) ─────────────────
        pos_path   = os.path.join(OUTPUT_DIR, "man_position_snapshot.csv")
        pos_fields = ["date", "und", "month", "group_id", "group_name", "view",
                       "lot_key", "instrument", "type", "side",
                       "lot_sz", "avg", "pnl_now", "pnl_sl", "pnl_target"]

        existing_pos: list[dict] = []
        if os.path.exists(pos_path):
            with open(pos_path, newline="", encoding="utf-8") as f:
                existing_pos = list(csv.DictReader(f))
        new_lot_keys = {
            (date, und, month, str(lot.get("lot_key", "")).strip())
            for g in groups for lot in g.get("lots", [])
        }
        kept_pos = [r for r in existing_pos
                    if (r["date"], r.get("und","NIFTY"), r["month"], r["lot_key"]) not in new_lot_keys]

        new_pos_rows = []
        for g in groups:
            for lot in g.get("lots", []):
                new_pos_rows.append({
                    "date":       date,
                    "und":        und,
                    "month":      month,
                    "group_id":   str(g.get("id",   "")).strip(),
                    "group_name": str(g.get("name", "")).strip(),
                    "view":       str(g.get("view", "")).strip(),
                    "lot_key":    str(lot.get("lot_key",    "")).strip(),
                    "instrument": str(lot.get("instrument", "")).strip(),
                    "type":       str(lot.get("type", "")).strip(),
                    "side":       str(lot.get("side", "")).strip(),
                    "lot_sz":     lot.get("lot_sz",     0),
                    "avg":        lot.get("avg",        0),
                    "pnl_now":    lot.get("pnl_now",    0),
                    "pnl_sl":     lot.get("pnl_sl",     ""),
                    "pnl_target": lot.get("pnl_target", ""),
                })

        with open(pos_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pos_fields)
            writer.writeheader()
            writer.writerows(kept_pos + new_pos_rows)

        # ── CSV 3: man_session_notes.csv (append-only, one row per save) ──────
        notes_path   = os.path.join(OUTPUT_DIR, "man_session_notes.csv")
        notes_fields = ["ts", "date", "und", "month", "nifty_spot", "banknifty_spot", "dte", "notes"]
        write_header = not os.path.exists(notes_path)
        with open(notes_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=notes_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date":          date,
                "und":           und,
                "month":         month,
                "nifty_spot":    nifty_spot     if nifty_spot     is not None else "",
                "banknifty_spot":banknifty_spot if banknifty_spot is not None else "",
                "dte":           dte            if dte            is not None else "",
                "notes":         notes,
            })

        return {
            "status": "ok",
            "rows_written": rows_written,
            "total_rows":   total_rows,
            "position_rows_written": len(new_pos_rows),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/man-action", tags=["Portfolio"],
          dependencies=[Depends(_require_portfolio_key)])
def post_man_action(payload: dict):
    """
    Append one drag-drop / remove event to rita_output/man_action_log.csv.

    This is the primary training-signal dataset for future ML/RAG models —
    captures every expert decision in its market context.

    Payload::

        {
          "date": "29-Mar-2026", "month": "APR",
          "action": "assign",           // "assign" | "unassign" | "remove"
          "lot_key": "NIFTY25APR22500CE_L1",
          "from_group": "",             // "" = pool
          "to_group":   "anchor",       // "" = pool / remove
          "nifty_spot": 22400.50,       // optional
          "banknifty_spot": 48200.0     // optional
        }
    """
    try:
        action  = str(payload.get("action",  "")).strip()
        lot_key = str(payload.get("lot_key", "")).strip()
        if not action or not lot_key:
            raise HTTPException(status_code=422, detail="action and lot_key are required")

        log_path   = os.path.join(OUTPUT_DIR, "man_action_log.csv")
        fieldnames = ["ts", "date", "month", "action", "lot_key",
                      "from_group", "to_group", "nifty_spot", "banknifty_spot"]
        write_header = not os.path.exists(log_path)

        nspot = payload.get("nifty_spot")
        bspot = payload.get("banknifty_spot")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date":          str(payload.get("date",       "")).strip(),
                "month":         str(payload.get("month",      "")).strip().upper(),
                "action":        action,
                "lot_key":       lot_key,
                "from_group":    str(payload.get("from_group", "")).strip(),
                "to_group":      str(payload.get("to_group",   "")).strip(),
                "nifty_spot":    nspot if nspot is not None else "",
                "banknifty_spot":bspot if bspot is not None else "",
            })

        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/man-pnl-history", tags=["Portfolio"],
         dependencies=[Depends(_require_portfolio_key)])
def get_man_pnl_history(month: str = "", und: str = ""):
    """
    Return manoeuvre P&L history from rita_output/man_pnl_history.csv.

    Optional ?month=APR&und=NIFTY filter.  Returns rows grouped by date, each
    with a list of group snapshots — ready for sparkline rendering.
    """
    try:
        csv_path = os.path.join(OUTPUT_DIR, "man_pnl_history.csv")
        if not os.path.exists(csv_path):
            return {"days": []}

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if month:
            rows = [r for r in rows if r["month"].upper() == month.upper()]
        if und:
            rows = [r for r in rows if r.get("und","NIFTY").upper() == und.upper()]

        # Group by date preserving insertion order
        by_date: dict = {}
        for r in rows:
            d = r["date"]
            if d not in by_date:
                by_date[d] = []
            by_date[d].append({
                "id":         r["group_id"],
                "name":       r["group_name"],
                "view":       r["view"],
                "pnl_now":    float(r["pnl_now"])    if r["pnl_now"]    not in ("", None) else None,
                "sl_pnl":     float(r["sl_pnl"])     if r["sl_pnl"]     not in ("", None) else None,
                "target_pnl": float(r["target_pnl"]) if r["target_pnl"] not in ("", None) else None,
                "lot_count":  int(r["lot_count"])    if r["lot_count"]   not in ("", None) else 0,
            })

        days = [{"date": d, "groups": gs} for d, gs in by_date.items()]
        return {"days": days}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User's natural-language investment question")
    portfolio_inr: float = Field(default=1_000_000, ge=1, description="Portfolio size in INR for stress/comparison scenarios")


@app.post("/api/v1/chat/warmup", tags=["Chat"], status_code=202)
def chat_warmup():
    """
    Pre-warm the classifier (loads SentenceTransformer + builds seed index).
    Called by the dashboard when the user opens Market Analysis so the first
    chat message is not delayed by model loading.
    Idempotent — safe to call multiple times.
    """
    from rita.core.classifier import _build_seed_index
    _build_seed_index()
    return {"status": "ready"}


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

    # Also log to mcp_call_log.csv so MCP Calls page reflects chat activity
    import csv as _csv
    _mcp_log_path = os.path.join(OUTPUT_DIR, "mcp_call_log.csv")
    _write_header = not os.path.exists(_mcp_log_path)
    with open(_mcp_log_path, "a", newline="", encoding="utf-8") as _f:
        _w = _csv.writer(_f)
        if _write_header:
            _w.writerow(["timestamp", "tool_name", "status", "duration_ms", "args_summary", "result_summary"])
        _w.writerow([
            __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "chat",
            status,
            round(latency_ms, 1),
            f"intent={result.intent.name}, conf={round(result.confidence, 3)}",
            response_text[:200],
        ])

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
