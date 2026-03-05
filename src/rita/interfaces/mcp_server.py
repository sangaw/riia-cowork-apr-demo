"""
RITA Interface — MCP Server (Claude Desktop / Cowork)
8 MCP tools, one per workflow step.
Each tool calls WorkflowOrchestrator and returns JSON via TextContent.
"""

import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import BACKTEST_START

# ─── Configuration ────────────────────────────────────────────────────────────

def _load_env():
    """Load .env file if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_env()

NIFTY_CSV_PATH = os.getenv(
    "NIFTY_CSV_PATH",
    r"C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\raw-data\nifty\merged.csv",
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./rita_output")

app = Server("rita")
_orchestrator: WorkflowOrchestrator = None


def _get_orchestrator() -> WorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator(NIFTY_CSV_PATH, OUTPUT_DIR)
    return _orchestrator


def _ok(step: int, name: str, result: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(
        {"step": step, "name": name, "status": "ok", "result": result},
        indent=2, default=str
    ))]


def _err(step: int, name: str, error: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(
        {"step": step, "name": name, "status": "error", "error": error},
        indent=2
    ))]


# ─── Tool Definitions ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="step1_set_financial_goal",
            description=(
                "Step 1 of 8 — Set financial goal anchored to real Nifty 50 historical returns. "
                "Validates target against 25-year CAGR and provides feasibility assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_return_pct": {
                        "type": "number",
                        "description": "Target total return percentage (e.g. 15.0 for 15%)"
                    },
                    "time_horizon_days": {
                        "type": "integer",
                        "description": "Investment horizon in days (e.g. 365 for 1 year)"
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["conservative", "moderate", "aggressive"],
                        "description": "Investor risk profile"
                    }
                },
                "required": ["target_return_pct", "time_horizon_days", "risk_tolerance"]
            }
        ),
        Tool(
            name="step2_analyze_market",
            description=(
                "Step 2 of 8 — Analyze current Nifty 50 market conditions using the latest "
                "252 trading days from the CSV. Returns trend, RSI, MACD, Bollinger Band "
                "position, ATR-derived sentiment, and EMA cross."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="step3_design_strategy",
            description=(
                "Step 3 of 8 — Design an investment strategy based on market research and "
                "financial goal. Returns allocation approach, rebalancing frequency, and "
                "constraint targets (Sharpe > 1, max drawdown < 10%)."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="step4_train_model",
            description=(
                "Step 4 of 8 — Train the Double DQN (DDQN) reinforcement learning model on "
                "Nifty 50 data from 2010-2022, then validate on 2023-2024 data. "
                "Reports training metrics and out-of-sample Sharpe ratio and max drawdown."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "timesteps": {
                        "type": "integer",
                        "description": "Training timesteps (default 200000, reduce to 50000 for quick test)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="step5_set_simulation_period",
            description=(
                "Step 5 of 8 — Set the backtest simulation period. "
                "Default: 2025-01-01 to the latest date in the dataset. "
                "Must be 2023-01-01 or later (cannot overlap training data)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": f"Start date (YYYY-MM-DD, default: {BACKTEST_START})"
                    },
                    "end": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD, default: latest in dataset)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="step6_run_backtest",
            description=(
                "Step 6 of 8 — Run the trained DDQN model through the simulation period "
                "day by day. Returns daily portfolio vs benchmark performance."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="step7_get_results",
            description=(
                "Step 7 of 8 — Generate the full performance report with interpretability plots. "
                "Returns Sharpe ratio, max drawdown, CAGR, win rate vs benchmark, "
                "and paths to 5 saved charts: returns, drawdown, action timeline, "
                "Q-value feature buckets, rolling Sharpe."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="step8_update_goal",
            description=(
                "Step 8 of 8 — Compare backtest results against the original financial goal "
                "and produce a revised, realistic goal for the next cycle. "
                "Closes the reinforcement learning feedback loop."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]


# ─── Tool Handlers ─────────────────────────────────────────────────────────────

HANDLERS = {
    "step1_set_financial_goal": None,
    "step2_analyze_market": None,
    "step3_design_strategy": None,
    "step4_train_model": None,
    "step5_set_simulation_period": None,
    "step6_run_backtest": None,
    "step7_get_results": None,
    "step8_update_goal": None,
}


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    orch = _get_orchestrator()
    args = arguments or {}

    try:
        if name == "step1_set_financial_goal":
            result = orch.step1_set_goal(
                float(args["target_return_pct"]),
                int(args["time_horizon_days"]),
                str(args["risk_tolerance"]),
            )
            return _ok(1, "Set Financial Goal", result["result"])

        elif name == "step2_analyze_market":
            result = orch.step2_analyze_market()
            return _ok(2, "Analyze Market Conditions", result["result"])

        elif name == "step3_design_strategy":
            result = orch.step3_design_strategy()
            return _ok(3, "Design Strategy", result["result"])

        elif name == "step4_train_model":
            timesteps = int(args.get("timesteps", 200_000))
            result = orch.step4_train_model(timesteps=timesteps)
            return _ok(4, "Train DDQN Model", result["result"])

        elif name == "step5_set_simulation_period":
            result = orch.step5_set_simulation_period(
                start=args.get("start", BACKTEST_START),
                end=args.get("end"),
            )
            return _ok(5, "Set Simulation Period", result["result"])

        elif name == "step6_run_backtest":
            result = orch.step6_run_backtest()
            return _ok(6, "Run Backtest", result["result"])

        elif name == "step7_get_results":
            result = orch.step7_get_results()
            # Return performance + plot paths (not raw time-series — too large for MCP)
            mcp_result = {
                "performance": result["result"]["performance"],
                "constraint_check": result["result"]["constraint_check"],
                "plots_saved_to": result["result"]["plots"],
                "output_dir": result["result"]["output_dir"],
            }
            return _ok(7, "Get Results", mcp_result)

        elif name == "step8_update_goal":
            result = orch.step8_update_goal()
            return _ok(8, "Update Financial Goal", result["result"])

        else:
            return _err(0, name, f"Unknown tool: {name}")

    except Exception as e:
        step_map = {
            "step1_set_financial_goal": 1, "step2_analyze_market": 2,
            "step3_design_strategy": 3, "step4_train_model": 4,
            "step5_set_simulation_period": 5, "step6_run_backtest": 6,
            "step7_get_results": 7, "step8_update_goal": 8,
        }
        return _err(step_map.get(name, 0), name, str(e))


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import asyncio
    asyncio.run(_async_main())


async def _async_main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    main()
