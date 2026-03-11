# RITA — Risk Informed Trading Approach
**Nifty 50 RL-powered investment system with Claude Desktop MCP integration**

RITA uses a Double DQN reinforcement learning model trained on 13 years of Nifty 50 data to make risk-aware allocation decisions (HOLD / 50% / 100%). It integrates with Claude Desktop via MCP, exposing 14 conversational tools that cover the full investment analysis cycle — from return expectations through performance feedback.

---

## Architecture

```
Claude Desktop (Cowork)
        │ MCP protocol
        ▼
src/rita/interfaces/mcp_server.py  — 14 MCP tools
        │
        ├── Core layer
        │   ├── data_loader.py         — Nifty 50 CSV loader + return estimate engine
        │   ├── technical_analyzer.py  — RSI, MACD, BB, ATR, EMA indicators + sentiment scoring
        │   ├── strategy_engine.py     — Rule-based HOLD/HALF/FULL proxy (mirrors RL action space)
        │   ├── rl_agent.py            — Double DQN training + episode runner (stable-baselines3)
        │   ├── performance.py         — Sharpe, MDD, portfolio comparison, stress test, feedback
        │   ├── risk_engine.py         — VaR timeline, trade events, drawdown budget
        │   ├── shap_explainer.py      — SHAP DeepExplainer on DQN Q-network
        │   └── training_tracker.py    — Per-round training history (CSV)
        │
        ├── Orchestration layer
        │   ├── workflow.py            — 8-step pipeline runner
        │   ├── session.py             — CSV state persistence
        │   └── monitor.py             — Phase timing
        │
        └── Interface layer
            ├── mcp_server.py          — MCP stdio server (14 tools)
            ├── streamlit_app.py       — 10-tab Streamlit UI
            └── rest_api.py            — FastAPI (13 endpoints)
```

---

## MCP Tools (14)

### Standalone Conversational Tools
| Tool | Trigger | What it returns |
|------|---------|----------------|
| `get_return_estimates` | "What returns can I expect from Nifty 50?" | 5 historical percentile scenarios (conservative→best-case), win rate, suggested target |
| `get_market_sentiment` | "How is the market looking?" | BULLISH/NEUTRAL/BEARISH from 5 signals (EMA, MACD, RSI, BB, ATR), score −6 to +6 |
| `get_strategy_recommendation` | "What allocation should I take?" | HOLD / HALF / FULL — mirrors RL action space, with rationale and override rules |
| `get_portfolio_scenarios` | "I have 10 lakh INR, show me scenarios" | Conservative/Moderate/Aggressive vs RITA model in INR (Sharpe, MDD, return) |
| `get_stress_scenarios` | "What if market moves 20%?" | ±% stress test across all profiles + RITA→HOLD protection row |
| `get_performance_feedback` | "How did RITA perform in 2025?" | Return %, CAGR, Sharpe, MDD, trade count, time allocation, realistic expectations |

### 8-Step Pipeline Tools
`step1_set_financial_goal` → `step2_analyze_market` → `step3_design_strategy` → `step4_train_model` → `step5_set_simulation_period` → `step6_run_backtest` → `step7_get_results` → `step8_update_goal`

---

## Performance (2025 Backtest — Apr–Dec)

| Metric | RITA | Buy & Hold |
|--------|------|-----------|
| Total return | 13.85% | 16.65% |
| CAGR | 19.79% | 23.92% |
| Sharpe ratio | **1.191** ✅ | ~0.9 implied |
| Max drawdown | **-4.55%** ✅ | -8%+ |
| Trades | 33 (18 buys, 15 sells) | — |

Constraints: Sharpe > 1.0 ✅ | Max drawdown < 10% ✅

---

## Quick Start

### Prerequisites
- Python 3.11 in `C:\Users\Sandeep\pyenv-envs\poc` (or your env with rita package installed)
- Nifty 50 OHLCV CSV at path set in `NIFTY_CSV_PATH`

### Run the Streamlit UI
```powershell
cd C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo
. .\activate-env.ps1
python run_ui.py
```
Opens at http://localhost:8501

### Run the pipeline headlessly
```powershell
python run_pipeline.py
```

### Run tests
```powershell
pytest tests/ -v
```

---

## Claude Desktop Setup (Windows)

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

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

Restart Claude Desktop. Tools appear automatically in conversations.

### Demo Conversation Flow
1. "What returns can I expect from Nifty 50 over 1 year?"
2. "How is the market looking right now?"
3. "What allocation should I take?"
4. "I have 10 lakh INR — show me scenarios for 2025"
5. "What if the market moves 20% in either direction?"
6. "How did the RITA model perform in 2025? What are realistic return expectations?"

---

## Streamlit UI (10 Tabs)

| Tab | Contents |
|-----|---------|
| 🏠 Dashboard | KPI strip (8 metrics), constraint badges, goal summary |
| 📋 Steps | Interactive 8-step strip with expandable detail panels |
| 📈 Performance | Returns, drawdown, Sharpe, Q-values — 4-column card grid |
| 🛡️ Risk View | Risk timeline, VaR, trade impact, regime breakdown |
| 🔍 Explainability | SHAP global importance, beeswarm, radar, dependence |
| 📉 Training | Round history, Sharpe/MDD progression across training runs |
| 📥 Export | JSON, HTML, CSV downloads |
| 🔭 Observability | Drift detection, API latency, health checks |
| 🚀 DevOps | Docker, CI/CD status |
| 🔌 MCP Calls | Isolated refresh (🔄), call log, per-tool latency charts |

---

## Key Design Decisions

- **Standalone MCP tools** load CSV directly — no workflow session needed for conversational use
- **Tool descriptions** use "ALWAYS use this tool when..." pattern to ensure Claude Desktop makes real tool calls
- **Action space alignment**: `get_strategy_recommendation` uses the same HOLD/HALF/FULL (0/1/2) space as the DDQN model
- **MCP logging**: All 14 tools write to `rita_output/mcp_call_log.csv` after every call
- **Fragment refresh**: MCP Calls tab uses `@st.fragment` — isolated re-run without triggering full pipeline

---

## Disclaimer

Research and educational tool. Not financial advice. Past performance does not guarantee future results.
