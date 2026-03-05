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

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import BACKTEST_START

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ────────────────────────────────────────────────────────────

CSV_PATH = os.getenv(
    "NIFTY_CSV_PATH",
    r"C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\raw-data\nifty\merged.csv",
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./rita_output")

# ─── Singleton orchestrator ───────────────────────────────────────────────────

_orchestrator: Optional[WorkflowOrchestrator] = None


def get_orchestrator() -> WorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        if not os.path.exists(CSV_PATH):
            raise RuntimeError(f"Data CSV not found: {CSV_PATH}. Set NIFTY_CSV_PATH env var.")
        _orchestrator = WorkflowOrchestrator(CSV_PATH, OUTPUT_DIR)
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
    version="0.2.0",
    lifespan=lifespan,
)

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


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Service health check."""
    orch = get_orchestrator()
    return {
        "status": "ok",
        "csv_loaded": orch._raw_df is not None,
        "model_exists": os.path.exists(os.path.join(OUTPUT_DIR, "rita_ddqn_model.zip")),
        "output_dir": OUTPUT_DIR,
    }


@app.get("/progress", tags=["System"])
def progress():
    """Return pipeline progress summary."""
    return get_orchestrator().session.get_progress_summary()


@app.post("/reset", tags=["System"])
def reset():
    """Reset orchestrator session (clears in-memory state, keeps saved files)."""
    global _orchestrator
    _orchestrator = WorkflowOrchestrator(CSV_PATH, OUTPUT_DIR)
    return {"status": "reset", "message": "Orchestrator session cleared."}


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
