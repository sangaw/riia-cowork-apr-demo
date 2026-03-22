# RITA — Risk Informed Trading Approach
**Nifty 50 RL investment system · Double DQN + Dual Model + Regime Routing · v1.0.0**

RITA trains a Double DQN reinforcement learning model on 13 years of Nifty 50 data to make risk-aware daily allocation decisions (HOLD / 50% / 100%). It ships with two UIs and 14 Claude Desktop MCP tools.

---

## Two UIs

| UI | Audience | Port | Command |
|----|----------|------|---------|
| **HTML Dashboard** | Business / functional users | :8000 | `python run_api.py` |
| **Streamlit App** | Data scientists / model developers | :8501 | `python run_ui.py` |

```powershell
# Activate shared Python env first
. .\activate-env.ps1

python run_api.py   # → http://localhost:8000/dashboard/
python run_ui.py    # → http://localhost:8501
```

---

## Architecture

```
Claude Desktop (MCP)  ·  HTML Dashboard  ·  Streamlit UI  ·  REST API
                              │
              ┌───────────────┴───────────────┐
              │        ORCHESTRATION           │
              │  workflow · session · monitor  │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │            CORE               │
              │  data_loader · technical_     │
              │  analyzer · rl_agent ·        │
              │  performance · goal_engine ·  │
              │  strategy_engine · classifier │
              │  chat_monitor                 │
              └───────────────────────────────┘
```

**Key source files:**

| File | Purpose |
|------|---------|
| `src/rita/core/data_loader.py` | CSV load, train/val/test splits, bear episode extraction |
| `src/rita/core/technical_analyzer.py` | RSI, MACD, BB, ATR, EMA, trend score, regime detection |
| `src/rita/core/rl_agent.py` | NiftyTradingEnv + Double DQN training + backtest runners |
| `src/rita/core/performance.py` | Sharpe, MDD, CAGR, portfolio comparison, stress test |
| `src/rita/core/classifier.py` | Chat intent classifier (sentence-transformers, 20 intents) |
| `src/rita/core/chat_monitor.py` | CSV-based chat query logging |
| `src/rita/orchestration/workflow.py` | 8-step pipeline orchestrator |
| `src/rita/interfaces/rest_api.py` | FastAPI (24+ endpoints + static HTML dashboard) |
| `src/rita/interfaces/streamlit_app.py` | Streamlit web UI (11 tabs) |
| `src/rita/interfaces/mcp_server.py` | MCP stdio server (14 tools) |

---

## RL Model — v1.5 Dual Model + Regime Routing

### State Space (9 features)
`daily_return · rsi_14 · macd · bb_pct_b · trend_score · allocation · days_remaining · atr_norm · ema_ratio_norm`

Feature 9 (`ema_ratio_norm`) detects medium-term regime breaks: `clip((ema_26/ema_50 − 1) × 20, −3, 3)`

### Dual Model
| Model | File | Training | Purpose |
|-------|------|----------|---------|
| Bull | `rita_ddqn_model.zip` | 500k steps, n_seeds=5, 2010–2022 full set | Normal market — growth optimisation |
| Bear | `rita_ddqn_bear_model.zip` | 200k steps, max 3 seeds, correction episodes | Capital preservation in downturns |

### Regime Detection
`detect_regime()` → BEAR if `ema_26/ema_50 < 0.99` for 3+ consecutive days; otherwise BULL.

`run_regime_episode()` switches models day-by-day during backtests.

---

## 8-Step Workflow

| Step | Method | What it does |
|------|--------|-------------|
| 1 | `step1_set_financial_goal` | Set target return %, horizon, risk tolerance |
| 2 | `step2_analyze_market` | Compute indicators; classify trend + sentiment |
| 3 | `step3_design_strategy` | Pick allocation approach |
| 4 | `step4_train_model(model_type)` | Train bull / bear / both |
| 5 | `step5_set_simulation_period` | Set backtest window |
| 6 | `step6_run_backtest(backtest_mode)` | Run model; compute performance |
| 7 | `step7_get_results` | Generate plots; run constraint check |
| 8 | `step8_update_goal` | Compare actual vs target; revise goal |

---

## MCP Tools (14)

### Standalone Conversational Tools
| Tool | What it returns |
|------|----------------|
| `get_return_estimates` | Historical Nifty percentile scenarios (conservative → best-case) |
| `get_market_sentiment` | BULLISH / NEUTRAL / BEARISH from 5 signals (score −6 to +6) |
| `get_strategy_recommendation` | HOLD / HALF / FULL — mirrors RL action space |
| `get_portfolio_scenarios` | ₹ comparison: Conservative / Moderate / Aggressive vs RITA |
| `get_stress_scenarios` | Point-in-time stress test for ±market moves |
| `get_performance_feedback` | Full backtest summary + realistic forward expectations |

### 8-Step Pipeline Tools
`step1_set_financial_goal` → `step2_analyze_market` → `step3_design_strategy` → `step4_train_model` → `step5_set_simulation_period` → `step6_run_backtest` → `step7_get_results` → `step8_update_goal`

---

## HTML Dashboard (:8000)

```
http://localhost:8000/dashboard/          ← landing page (4 cards)
http://localhost:8000/dashboard/rita.html ← main RITA app
http://localhost:8000/dashboard/fno.html  ← FnO Portfolio Manager
http://localhost:8000/dashboard/ops.html  ← Operations Portal
```

**4-phase navigation in rita.html:**
- Phase 01 Plan (green): Data Prep, Financial Goal, Market Analysis, Market Signals, Strategy
- Phase 02 Build (orange): Train Model
- Phase 03 Analyse (blue): Performance, Trade Journal, Explainability
- Phase 04 Monitor (purple): Risk View, Training Progress, Observability, MCP Calls, Audit

**Market Analysis page** has a 3-panel chat interface (5fr 5fr 3fr):
- Panel 1: Market Analysis + Analyze Market button
- Panel 2: Converse with RITA Agent (intent-based, deterministic — no live Claude API)
- Panel 3: 10 suggested question chips

---

## Streamlit App (:8501)

11 tabs after pipeline run: Dashboard · Steps · Performance · Risk View · Trade Journal · Explainability · Training · Export · Observability · DevOps · MCP Calls

Pipeline buttons: **🏗 Build Model Pipeline** (`force_retrain=True`) | **▶ Re-use Model Pipeline** (`force_retrain=False`)

---

## Performance

Current backtest (Apr 2025 – Feb 2026, 221 days, 8-feature model):

| Metric | RITA | Buy & Hold |
|--------|------|-----------|
| Return | 12.0% | 12.41% |
| Sharpe | 0.826 | — |
| Max drawdown | −3.49% | — |
| Trades | 111 | — |

*9-feature bull + bear models not yet trained. Retrain expected to improve Sharpe.*

Constraints: Sharpe > 1.0 | Max drawdown < 10%

---

## Quick Start

```powershell
# 1. Activate environment
. .\activate-env.ps1

# 2. Quick check (no training, ~10s)
python verify.py

# 3. Full pipeline (reuses saved model, ~30s)
python run_pipeline.py

# 4. Run tests
pytest tests/                          # all 39 tests
pytest tests/ -k "not APIEndpoints"   # unit tests only (no CSV needed)
```

---

## Claude Desktop MCP Config

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rita-cowork": {
      "command": "C:\\path\\to\\your\\python.exe",
      "args": ["-m", "rita.interfaces.mcp_server"],
      "env": {
        "NIFTY_CSV_PATH": "C:\\path\\to\\nifty_merged.csv",
        "OUTPUT_DIR": "C:\\path\\to\\rita-cowork-demo\\rita_output",
        "PYTHONPATH": "C:\\path\\to\\rita-cowork-demo\\src",
        "PYTHON_ENV": "development"
      }
    }
  }
}
```

Restart Claude Desktop. RITA tools appear automatically.

### Demo Conversation
1. "What returns can I expect from Nifty 50 over 1 year?"
2. "How is the market looking right now?"
3. "What allocation should I take?"
4. "I have 10 lakh INR — show me scenarios"
5. "What if the market moves 20% in either direction?"
6. "How did RITA perform? What are realistic expectations going forward?"

---

## Data

- Source: Local CSV — no external API calls
- Instrument: Nifty 50 index (daily OHLCV)
- Current data: 6,650 rows through 20-Mar-2026
- Training: 2010–2022 | Validation: 2023–2024 | Backtest: 2025–present
- Input dir: `rita_input/` — drop NSE-format CSVs, run Data Prep

---

## Tech Stack

stable-baselines3 · gymnasium · PyTorch · pandas · numpy · ta · FastAPI · uvicorn · streamlit · plotly · sentence-transformers · pytest · hatchling

---

*Research and educational tool. Not financial advice.*
