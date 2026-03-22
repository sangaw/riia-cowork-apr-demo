"""
RITA Interface — MCP Server (Claude Desktop / Cowork)
8 MCP tools, one per workflow step.
Each tool calls WorkflowOrchestrator and returns JSON via TextContent.
"""

import csv
import json
import os
import time
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import (
    BACKTEST_START,
    load_nifty_csv,
    parse_period_to_days,
    get_period_return_estimates,
)
from rita.core.technical_analyzer import calculate_indicators, get_market_summary, get_sentiment_score
from rita.core.strategy_engine import get_allocation_recommendation
from rita.core.performance import build_portfolio_comparison, simulate_stress_scenarios, build_performance_feedback
from rita.config import NIFTY_CSV_PATH, OUTPUT_DIR, TRAIN_TIMESTEPS

# ─── Configuration ────────────────────────────────────────────────────────────

MCP_LOG_PATH = os.path.join(OUTPUT_DIR, "mcp_call_log.csv")

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


def _log_mcp_call(
    tool_name: str,
    status: str,
    duration_ms: float,
    args_summary: str,
    result_summary: str,
) -> None:
    """Append one row to mcp_call_log.csv after every tool invocation."""
    log_path = os.path.join(OUTPUT_DIR, "mcp_call_log.csv")
    write_header = not os.path.exists(log_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "tool_name", "status",
                "duration_ms", "args_summary", "result_summary",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            tool_name,
            status,
            round(duration_ms, 1),
            args_summary[:120],
            result_summary[:200],
        ])


# ─── Tool Definitions ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_return_estimates",
            description=(
                "ALWAYS use this tool when the user asks about Nifty 50 returns, expected "
                "returns, investment planning, how much they can earn, or what return to "
                "expect over any period. "
                "Returns REAL data-driven Nifty 50 return estimates computed from actual "
                "25+ years of historical price data — NOT from general knowledge. "
                "Accepts an optional period (e.g. '1y', '6m', '3y'); defaults to '1y'. "
                "Returns 5 scenarios: conservative (10th pct), cautious (25th), median (50th), "
                "optimistic (75th), best_case (90th) — all grounded in rolling historical windows."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": (
                            "Investment horizon. Examples: '1d', '2w', '3m', '6m', '1y', '2y', '3y'. "
                            "Also accepts plain days as a string e.g. '90'. Defaults to '1y'."
                        ),
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_market_sentiment",
            description=(
                "ALWAYS use this tool when the user asks about market sentiment, market conditions, "
                "how the market looks, whether it is a good time to invest, market mood, or "
                "current Nifty 50 direction. "
                "Returns a consolidated sentiment rating (BULLISH / CAUTIOUSLY_BULLISH / NEUTRAL / "
                "CAUTIOUSLY_BEARISH / BEARISH) computed from live indicator signals — "
                "trend (EMA cross), MACD, RSI, Bollinger Bands, and volatility (ATR). "
                "Standalone — no prior steps needed. "
                "Optional: lookback_days (default 252 = 1 trading year)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lookback_days": {
                        "type": "integer",
                        "description": "Number of recent trading days to analyse (default 252 = 1 year).",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_strategy_recommendation",
            description=(
                "ALWAYS use this tool when the user asks what allocation to take, whether to "
                "invest, how much to invest in Nifty 50, what strategy to follow, or whether "
                "to hold/buy/reduce position. "
                "Returns a HOLD / HALF (50%) / FULL (100%) recommendation using the same "
                "action space as the RITA DDQN model, calibrated toward Sharpe > 1.0 and "
                "max drawdown < 10%. Includes rationale, primary constraint driving the "
                "decision, and upgrade/downgrade triggers. "
                "Standalone — no prior steps needed. "
                "Optional: lookback_days (default 252)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lookback_days": {
                        "type": "integer",
                        "description": "Recent trading days window for indicator calculation (default 252).",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_portfolio_scenarios",
            description=(
                "ALWAYS use this tool when the user mentions a portfolio amount in INR and asks "
                "about scenarios, projections, or how their money would perform. "
                "Compares three manual fixed-allocation strategies (Conservative 30%, "
                "Moderate 60%, Aggressive 100%) against the RITA RL model on the same "
                "historical period and Nifty price data. "
                "Returns final portfolio values in INR, return %, Sharpe ratio, and max "
                "drawdown for each — showing where the RL model stands vs manual approaches. "
                "Requires the RITA model to have been run at least once (backtest_daily.csv "
                "must exist). "
                "Optional: portfolio_inr (default 1000000 = Rs 10 lakh)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "portfolio_inr": {
                        "type": "number",
                        "description": "Starting capital in INR (default 1000000 = Rs 10 lakh).",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_stress_scenarios",
            description=(
                "ALWAYS use this tool when the user asks what happens if the market moves up or "
                "down by a certain percentage, stress test scenarios, crash scenarios, bull run "
                "scenarios, or portfolio impact of market moves. "
                "Simulates Conservative (30%), Moderate (60%), Aggressive (100%), and RITA RL "
                "model at current recommendation — showing final portfolio value in INR, "
                "profit/loss, drawdown, and whether the 10% drawdown constraint is breached. "
                "Also shows RITA -> HOLD (0%) as the model's downgrade protection. "
                "Standalone — no prior steps needed. "
                "Optional: portfolio_inr (default 1000000), "
                "market_moves as comma-separated list e.g. '-20,-10,10,20' (default)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "portfolio_inr": {
                        "type": "number",
                        "description": "Starting capital in INR (default 1000000 = Rs 10 lakh).",
                    },
                    "market_moves": {
                        "type": "string",
                        "description": (
                            "Comma-separated market move percentages to simulate. "
                            "E.g. '-20,-10,10,20' or '-30,30'. Default: '-20,-10,-5,5,10,20'."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_performance_feedback",
            description=(
                "ALWAYS use this tool when the user asks how the RITA model performed, what "
                "returns it generated, what the Sharpe ratio was, how many trades it took, "
                "what are realistic return expectations, or asks for a performance summary "
                "or outcome analysis for 2025. "
                "Returns a comprehensive outcome report: period, total return %, CAGR, Sharpe "
                "ratio, max drawdown, trade activity (buys/sells/holds), time allocation "
                "breakdown, training round info, and realistic forward return expectations. "
                "Requires the RITA model to have been run at least once (backtest_daily.csv "
                "and performance_summary.csv must exist in the output folder)."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
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
    t0 = time.time()

    def _ms() -> float:
        return (time.time() - t0) * 1000

    try:
        if name == "get_performance_feedback":
            import pandas as pd
            backtest_path = os.path.join(OUTPUT_DIR, "backtest_daily.csv")
            perf_path     = os.path.join(OUTPUT_DIR, "performance_summary.csv")
            history_path  = os.path.join(OUTPUT_DIR, "training_history.csv")

            missing = [p for p in [backtest_path, perf_path] if not os.path.exists(p)]
            if missing:
                msg = (
                    "RITA backtest data not found. "
                    "Please run the model first via the Streamlit UI (Run Pipeline button) "
                    "or call steps 1–7 via MCP."
                )
                _log_mcp_call(name, "error", _ms(), "", msg)
                return [TextContent(type="text", text=json.dumps(
                    {"tool": "get_performance_feedback", "status": "error", "error": msg},
                    indent=2,
                ))]

            backtest_df = pd.read_csv(backtest_path)
            perf_df     = pd.read_csv(perf_path)
            history_df  = pd.read_csv(history_path) if os.path.exists(history_path) else None

            feedback = build_performance_feedback(backtest_df, perf_df, history_df)
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"period={feedback.get('period', {}).get('start')} to "
                f"{feedback.get('period', {}).get('end')}, "
                f"return={feedback.get('return_metrics', {}).get('total_return_pct')}%, "
                f"sharpe={feedback.get('risk_metrics', {}).get('sharpe_ratio')}, "
                f"trades={feedback.get('trade_activity', {}).get('total_trades')}",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_performance_feedback", "status": "ok", "result": feedback},
                indent=2, default=str,
            ))]

        elif name == "get_market_sentiment":
            lookback = int(args.get("lookback_days", 252))
            df_raw = load_nifty_csv(NIFTY_CSV_PATH)
            df_feat = calculate_indicators(df_raw)
            window = df_feat.iloc[-lookback:]
            summary = get_market_summary(window)
            scored = get_sentiment_score(summary)
            result = {
                "as_of_date": summary["date"],
                "close": summary["close"],
                "lookback_days": lookback,
                "overall_sentiment": scored["overall_sentiment"],
                "signal_summary": scored["signal_summary"],
                "signals": scored["signals"],
                "score": f"{scored['total_score']:+d} / {scored['max_score']}",
                "raw_indicators": summary,
            }
            _log_mcp_call(
                name, "ok", _ms(),
                f"lookback={lookback}",
                f"sentiment={scored['overall_sentiment']}, score={scored['total_score']:+d}, "
                f"trend={summary['trend']}, rsi={summary['rsi_14']:.1f}, macd={summary['macd_signal']}",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_market_sentiment", "status": "ok", "result": result},
                indent=2, default=str,
            ))]

        elif name == "get_stress_scenarios":
            portfolio_inr = float(args.get("portfolio_inr", 1_000_000))
            moves_str     = args.get("market_moves", "-20,-10,-5,5,10,20")
            market_moves  = [int(x.strip()) for x in moves_str.split(",")]

            # Auto-derive current RITA allocation from live market signals
            df_raw  = load_nifty_csv(NIFTY_CSV_PATH)
            df_feat = calculate_indicators(df_raw)
            summary = get_market_summary(df_feat.iloc[-252:])
            scored  = get_sentiment_score(summary)
            rec     = get_allocation_recommendation(summary, scored)
            rita_alloc_pct = float(rec["allocation_pct"])

            result = simulate_stress_scenarios(portfolio_inr, market_moves, rita_alloc_pct)
            result["as_of_date"]          = summary["date"]
            result["market_sentiment"]    = scored["overall_sentiment"]
            result["rita_action"]         = rec["recommendation"]
            result["rita_rationale"]      = rec["rationale"]

            _log_mcp_call(
                name, "ok", _ms(),
                f"portfolio_inr={int(portfolio_inr)}, moves={moves_str}",
                f"rita_rec={rec['recommendation']} ({int(rita_alloc_pct)}%), "
                f"sentiment={scored['overall_sentiment']}, "
                f"worst_case={min(market_moves):+d}%",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_stress_scenarios", "status": "ok", "result": result},
                indent=2, default=str,
            ))]

        elif name == "get_portfolio_scenarios":
            portfolio_inr = float(args.get("portfolio_inr", 1_000_000))
            backtest_path = os.path.join(OUTPUT_DIR, "backtest_daily.csv")
            if not os.path.exists(backtest_path):
                msg = (
                    "RITA model backtest data not found. "
                    "Please run the model first via the Streamlit UI (Run Pipeline button) "
                    "or call step6_run_backtest."
                )
                _log_mcp_call(name, "error", _ms(), f"portfolio_inr={portfolio_inr}", msg)
                return [TextContent(type="text", text=json.dumps(
                    {"tool": "get_portfolio_scenarios", "status": "error", "error": msg},
                    indent=2,
                ))]
            import pandas as pd
            backtest_df = pd.read_csv(backtest_path)
            comparison  = build_portfolio_comparison(backtest_df, portfolio_inr)
            _log_mcp_call(
                name, "ok", _ms(),
                f"portfolio_inr={int(portfolio_inr)}",
                f"period={comparison['period_start']} to {comparison['period_end']}, "
                f"nifty={comparison['nifty_return_pct']}%, "
                f"rita_sharpe={comparison['profiles']['rita_model']['sharpe_ratio']}, "
                f"winner={comparison['sharpe_winner']}",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_portfolio_scenarios", "status": "ok", "result": comparison},
                indent=2, default=str,
            ))]

        elif name == "get_strategy_recommendation":
            lookback = int(args.get("lookback_days", 252))
            df_raw  = load_nifty_csv(NIFTY_CSV_PATH)
            df_feat = calculate_indicators(df_raw)
            window  = df_feat.iloc[-lookback:]
            summary = get_market_summary(window)
            scored  = get_sentiment_score(summary)
            rec     = get_allocation_recommendation(summary, scored)
            result  = {
                "as_of_date":         summary["date"],
                "close":              summary["close"],
                "lookback_days":      lookback,
                "recommendation":     rec["recommendation"],
                "allocation_pct":     rec["allocation_pct"],
                "action_code":        rec["action_code"],
                "primary_constraint": rec["primary_constraint"],
                "rationale":          rec["rationale"],
                "upgrade_trigger":    rec["upgrade_trigger"],
                "downgrade_trigger":  rec["downgrade_trigger"],
                "override_applied":   rec["override_applied"],
                "override_reason":    rec["override_reason"],
                "market_sentiment":   scored["overall_sentiment"],
                "sentiment_score":    f"{scored['total_score']:+d} / 6",
            }
            _log_mcp_call(
                name, "ok", _ms(),
                f"lookback={lookback}",
                f"recommendation={rec['recommendation']} ({rec['allocation_pct']}%), "
                f"sentiment={scored['overall_sentiment']}, "
                f"constraint={rec['primary_constraint']}, "
                f"override={rec['override_applied']}",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_strategy_recommendation", "status": "ok", "result": result},
                indent=2, default=str,
            ))]

        elif name == "get_return_estimates":
            period_str = args.get("period", "1y")
            period_days = parse_period_to_days(period_str)
            df = load_nifty_csv(NIFTY_CSV_PATH)
            estimates = get_period_return_estimates(df, period_days)
            _log_mcp_call(
                name, "ok", _ms(),
                f"period={period_str}",
                f"windows={estimates['sample_windows']}, "
                f"conservative={estimates['suggested_target_pct']}%, "
                f"median={estimates['scenarios']['median']['total_return_pct']}%, "
                f"win_rate={estimates['win_rate_pct']}%",
            )
            return [TextContent(type="text", text=json.dumps(
                {"tool": "get_return_estimates", "status": "ok", "result": estimates},
                indent=2, default=str,
            ))]

        elif name == "step1_set_financial_goal":
            result = orch.step1_set_goal(
                float(args["target_return_pct"]),
                int(args["time_horizon_days"]),
                str(args["risk_tolerance"]),
            )
            r = result["result"]
            _log_mcp_call(
                name, "ok", _ms(),
                f"target={args['target_return_pct']}%, horizon={args['time_horizon_days']}d, "
                f"risk={args['risk_tolerance']}",
                f"feasibility={r['feasibility']}, "
                f"annualised={r['annualized_target_pct']}%, "
                f"suggested={r['suggested_realistic_target_pct']}%",
            )
            return _ok(1, "Set Financial Goal", r)

        elif name == "step2_analyze_market":
            result = orch.step2_analyze_market()
            r = result["result"]
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"trend={r.get('trend')}, rsi={r.get('rsi_14', 0):.1f}, "
                f"macd={r.get('macd_signal', 'n/a')}, bb={r.get('bb_position', 'n/a')}",
            )
            return _ok(2, "Analyze Market Conditions", r)

        elif name == "step3_design_strategy":
            result = orch.step3_design_strategy()
            r = result["result"]
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"strategy={r.get('name')}, alloc={r.get('base_allocation_pct')}%, "
                f"rebalance={r.get('rebalancing_frequency')}",
            )
            return _ok(3, "Design Strategy", r)

        elif name == "step4_train_model":
            timesteps = int(args.get("timesteps", TRAIN_TIMESTEPS))
            result = orch.step4_train_model(timesteps=timesteps)
            r = result["result"]
            val = r.get("validation", {})
            _log_mcp_call(
                name, "ok", _ms(),
                f"timesteps={timesteps}",
                f"source={r.get('source')}, "
                f"val_sharpe={val.get('sharpe_ratio', 0):.3f}, "
                f"val_mdd={val.get('max_drawdown_pct', 0):.1f}%",
            )
            return _ok(4, "Train DDQN Model", r)

        elif name == "step5_set_simulation_period":
            result = orch.step5_set_simulation_period(
                start=args.get("start", BACKTEST_START),
                end=args.get("end"),
            )
            r = result["result"]
            _log_mcp_call(
                name, "ok", _ms(),
                f"start={args.get('start', BACKTEST_START)}, end={args.get('end', 'latest')}",
                f"period={r.get('start')} to {r.get('end')}, days={r.get('trading_days')}",
            )
            return _ok(5, "Set Simulation Period", r)

        elif name == "step6_run_backtest":
            result = orch.step6_run_backtest()
            r = result["result"]
            perf = r.get("performance", {})
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"return={perf.get('portfolio_total_return_pct', 0):.1f}%, "
                f"sharpe={perf.get('sharpe_ratio', 0):.3f}, "
                f"mdd={perf.get('max_drawdown_pct', 0):.1f}%",
            )
            return _ok(6, "Run Backtest", r)

        elif name == "step7_get_results":
            result = orch.step7_get_results()
            r = result["result"]
            perf = r.get("performance", {})
            mcp_result = {
                "performance": perf,
                "constraint_check": r.get("constraint_check"),
                "plots_saved_to": r.get("plots"),
                "output_dir": r.get("output_dir"),
            }
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"sharpe={perf.get('sharpe_ratio', 0):.3f}, "
                f"cagr={perf.get('portfolio_cagr_pct', 0):.1f}%, "
                f"mdd={perf.get('max_drawdown_pct', 0):.1f}%, "
                f"plots={len(r.get('plots', []))}",
            )
            return _ok(7, "Get Results", mcp_result)

        elif name == "step8_update_goal":
            result = orch.step8_update_goal()
            r = result["result"]
            _log_mcp_call(
                name, "ok", _ms(), "",
                f"assessment={r.get('assessment')}, "
                f"revised_target={r.get('revised_target_pct')}%, "
                f"delta={r.get('goal_delta_pct')}%",
            )
            return _ok(8, "Update Financial Goal", r)

        else:
            _log_mcp_call(name, "error", _ms(), str(args)[:80], f"Unknown tool: {name}")
            return _err(0, name, f"Unknown tool: {name}")

    except Exception as e:
        _log_mcp_call(name, "error", _ms(), str(args)[:80], str(e)[:200])
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
