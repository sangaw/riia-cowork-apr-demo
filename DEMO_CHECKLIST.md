# RITA Demo Checklist

## What's Implemented

### Core ML System
- [x] Double DQN (DDQN) RL model trained on Nifty 50 data (2010–2022)
- [x] 7-feature state space: daily_return, RSI, MACD, Bollinger %B, trend_score, allocation, days_remaining
- [x] 3-action space: HOLD (0%), HALF (50%), FULL (100%)
- [x] Constraints: Sharpe > 1.0 AND max drawdown < 10%
- [x] Backtest period: 2025 (Apr–Dec), Sharpe 1.191, MDD -4.55%, Return 13.85%

### MCP Integration (14 tools)
- [x] **get_return_estimates** — historical Nifty 50 return scenarios (1d to 3y)
- [x] **get_market_sentiment** — BULLISH/NEUTRAL/BEARISH from 5 live signals (EMA, MACD, RSI, BB, ATR)
- [x] **get_strategy_recommendation** — HOLD/HALF/FULL aligned to RL model action space
- [x] **get_portfolio_scenarios** — Compare Conservative/Moderate/Aggressive vs RITA in INR
- [x] **get_stress_scenarios** — Stress test for market moves (e.g. ±10%, ±20%)
- [x] **get_performance_feedback** — Outcome analyst: full 2025 backtest summary + realistic expectations
- [x] **step1–step8** — Full 8-step pipeline via MCP

### Streamlit UI (10 tabs)
- [x] 🏠 Dashboard — KPI strip + constraint badges
- [x] 📋 Steps — interactive 8-step strip
- [x] 📈 Performance — returns, drawdown, Sharpe, Q-values
- [x] 🛡️ Risk View — risk timeline, drawdown budget, trade impact, regimes
- [x] 🔍 Explainability — SHAP global importance, beeswarm, radar, dependence
- [x] 📉 Training — round history, Sharpe/MDD/return trends
- [x] 📥 Export — JSON, HTML, CSV downloads
- [x] 🔭 Observability — drift detection, latency, API health
- [x] 🚀 DevOps — Docker, CI/CD status
- [x] 🔌 MCP Calls — isolated refresh, call log, per-tool latency

### Infrastructure
- [x] FastAPI REST API (13 endpoints) with pytest suite (39 tests)
- [x] Docker + GitHub CI/CD
- [x] SHAP explainability (DeepExplainer on DQN Q-network)
- [x] Risk engine (VaR, trade events, phase breakdown)
- [x] Training tracker (round history, val vs backtest metrics)

---

## Demo Conversation Flow (Claude Desktop)

Start a new chat in Claude Desktop — RITA tools are available automatically.

### Step 1 — Return Expectations
> "What returns can I expect from Nifty 50 over 1 year?"

→ `get_return_estimates`: Shows 5 historical percentile scenarios (conservative to best-case) with win rate and suggested target.

### Step 2 — Market Sentiment
> "How is the market looking right now?"

→ `get_market_sentiment`: Consolidated BULLISH/NEUTRAL/BEARISH from EMA cross, MACD, RSI, Bollinger, ATR. Score −6 to +6.

### Step 3 — Strategy Recommendation
> "What allocation should I take in Nifty 50?"

→ `get_strategy_recommendation`: HOLD / HALF (50%) / FULL (100%) — mirrors RL model action space. Includes rationale, override rules, upgrade/downgrade triggers.

### Step 4 — Portfolio Scenarios
> "I have 10 lakh INR. Show me how different strategies would have performed in 2025."

→ `get_portfolio_scenarios`: Conservative (30%), Moderate (60%), Aggressive (100%) vs RITA — final INR values, Sharpe, MDD, return for each.

### Step 5 — Stress Test
> "What if the market moves 20% up or down from here?"

→ `get_stress_scenarios`: Point-in-time stress across all profiles + RITA current + RITA→HOLD. Shows ₹ P&L, drawdown breach flag, and RITA's MDD protection mechanism.

### Step 6 — Performance Feedback
> "How did the RITA model perform in 2025? What are realistic return expectations going forward?"

→ `get_performance_feedback`: Full outcome report — return %, CAGR, Sharpe, MDD, 33 trades (18 buys, 15 sells), time at each allocation, constraint verdict, 1y/3y forward expectations.

---

## Claude Desktop Setup (Windows)

Config file: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rita-cowork": {
      "command": "C:\\Users\\Sandeep\\pyenv-envs\\poc\\Scripts\\python.exe",
      "args": ["-m", "rita.interfaces.mcp_server"],
      "env": {
        "NIFTY_CSV_PATH": "C:\\path\\to\\merged.csv",
        "OUTPUT_DIR": "C:\\path\\to\\rita-cowork-demo\\rita_output",
        "PYTHONPATH": "C:\\path\\to\\rita-cowork-demo\\src",
        "PYTHON_ENV": "development"
      }
    }
  }
}
```

After editing, restart Claude Desktop. RITA tools appear automatically in conversations.

---

## Launching the Streamlit UI

```powershell
cd C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo
. .\activate-env.ps1
python run_ui.py
```

Opens at http://localhost:8501. The 🔌 MCP Calls tab auto-refreshes via the 🔄 Refresh button (isolated — no full pipeline re-run).

---

## Key Architectural Decisions

- Standalone MCP tools (1–6) load CSV directly — no workflow session required
- Tool descriptions use "ALWAYS use this tool when..." to force Claude Desktop to make real calls (not answer from training knowledge)
- Strategy recommendation uses the same 3-action space as the DDQN model (HOLD/HALF/FULL)
- Stress test auto-derives current RITA allocation from live market signals
- All 14 tools log to `rita_output/mcp_call_log.csv` with timestamp, duration_ms, args, result summary
- MCP Calls tab uses `@st.fragment` (isolated re-run) — click 🔄 Refresh without triggering full pipeline

---

**Status**: Ready for Demo
**Last Updated**: March 2026
**Test Environment**: Windows 11 / PyEnv poc / Python 3.11
