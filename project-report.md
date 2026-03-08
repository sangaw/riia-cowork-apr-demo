# RITA — Project Report
**Risk Informed Trading Approach · Nifty 50 RL Investment System**

---

## 1. Project Overview

RITA is a Nifty 50 index investment system powered by a Reinforcement Learning (Double DQN) model. It follows an 8-step workflow — from goal setting and market analysis, through model training and backtesting, to goal feedback — and is callable through four interfaces: MCP (Claude Desktop), a Python client SDK, a Streamlit web UI, and a FastAPI REST API.

### Core idea
Given only historical Nifty 50 OHLCV data, train an RL agent to decide daily: hold cash (0%), go half-invested (50%), or fully invested (100%). Constrain the strategy to achieve Sharpe ratio > 1.0 and maximum drawdown < 10%.

---

## 2. Architecture

RITA follows a strict 3-layer architecture. No layer accesses a lower layer's internals — all communication goes through defined function calls.

```
┌───────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                               │
│  MCP Server (Claude Desktop) | Streamlit UI | Python SDK | REST API  │
└──────────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                         │
│   WorkflowOrchestrator  ·  SessionManager  ·  PhaseMonitor  │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│                     CORE LAYER                               │
│  DataLoader · TechnicalAnalyzer · RLAgent · Performance      │
│  GoalEngine · StrategyEngine                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
rita-cowork-demo/
├── src/rita/
│   ├── core/
│   │   ├── data_loader.py          # CSV load, splits, historical stats
│   │   ├── technical_analyzer.py   # RSI, MACD, Bollinger, ATR, EMA
│   │   ├── rl_agent.py             # NiftyTradingEnv + Double DQN
│   │   ├── performance.py          # Sharpe, MDD, CAGR, 5 matplotlib plots
│   │   ├── goal_engine.py          # Goal setting + feedback update
│   │   └── strategy_engine.py      # Constraint validation
│   ├── orchestration/
│   │   ├── workflow.py             # WorkflowOrchestrator (8-step runner)
│   │   ├── session.py              # SessionManager (CSV persistence)
│   │   └── monitor.py              # PhaseMonitor (timing + logging)
│   └── interfaces/
│       ├── mcp_server.py           # 8 MCP tools → WorkflowOrchestrator
│       ├── python_client.py        # RITAClient Python SDK
│       ├── streamlit_app.py        # Web UI (Phase 2)
│       └── rest_api.py             # FastAPI REST API (Phase 3)
├── tests/
│   └── test_core.py                # 39 pytest tests (unit + API integration)
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI (Python 3.11 + 3.12)
├── Dockerfile                      # Python 3.12-slim image
├── docker-compose.yml              # API (8000) + UI (8501) services
├── run_pipeline.py                 # CLI: full 8-step pipeline
├── run_ui.py                       # CLI: launch Streamlit UI (auto port)
├── run_api.py                      # CLI: launch FastAPI server
├── verify.py                       # Quick test: steps 1–3, no training
├── test_steps5to8.py               # Quick test: steps 5–8 with saved model
├── project-report.md               # This document
├── pyproject.toml                  # Package + dependencies
├── activate-env.ps1.example        # Template for activating shared Python env
└── config/
    └── claude_desktop_config.json.example  # MCP config template
```

---

## 4. The 8-Step Workflow

| Step | Method | What it does |
|------|--------|-------------|
| 1 | `step1_set_goal` | Set target return %, horizon, risk tolerance. Validates feasibility against real Nifty history |
| 2 | `step2_analyze_market` | Compute RSI, MACD, Bollinger, ATR, EMA on last 252 trading days. Classify trend + sentiment |
| 3 | `step3_design_strategy` | Pick allocation approach (0–100%) based on trend × risk tolerance matrix |
| 4 | `step4_train_model` | Train Double DQN on 2010–2022 data; validate on 2023–2024. Reuses saved model by default |
| 5 | `step5_set_simulation_period` | Set backtest window (default: 2025-01-01 → latest in CSV) |
| 6 | `step6_run_backtest` | Run trained model step-by-step on simulation period; compute performance |
| 7 | `step7_get_results` | Generate all 5 plots; run constraint check; return full report |
| 8 | `step8_update_goal` | Compare actual vs target return; revise goal for next cycle |

---

## 5. Data

| Property | Detail |
|----------|--------|
| Source | Local CSV: `merged.csv` (no external API calls) |
| Instrument | Nifty 50 index (daily OHLCV) |
| Coverage | 1999-07-01 → 2025-12-31 (6,594 rows) |
| Columns | `date, open, high, low, close, shares traded, turnover (₹ cr)` |
| Training split | 2010-01-01 → 2022-12-31 (3,226 rows) |
| Validation split | 2023-01-01 → 2024-12-31 (494 rows) |
| Backtest split | 2025-01-01 → latest (180 rows as of Dec 2025) |

### Nifty 50 Historical Stats (full history)
| Metric | Value |
|--------|-------|
| CAGR | 12.56% |
| Sharpe ratio | 0.331 |
| Max drawdown | −59.86% |
| Best year | +75.76% |
| Worst year | −51.79% |

---

## 6. Reinforcement Learning Model

### Algorithm: Double DQN
Implemented via **stable-baselines3 DQN** with `target_update_interval=1000`.
Double DQN reduces Q-value overestimation by separating action selection (online network) from action evaluation (target network).

### Environment: `NiftyTradingEnv`

| Component | Detail |
|-----------|--------|
| State space | 7 features (see below) |
| Action space | Discrete(3): 0% / 50% / 100% invested |
| Episode length | 252 trading days (random window from training data) |
| Reward | `portfolio_return − 2.0` if drawdown > 10%, else `portfolio_return + 0.001` if positive |

**State features (7-dimensional):**

| Feature | Normalisation |
|---------|--------------|
| Daily return | × 10, clipped ±3 |
| RSI-14 | ÷ 100 → [0, 1] |
| MACD | z-score ÷ (3 × std), clipped ±3 |
| Bollinger %B | clipped [−0.5, 1.5] |
| Trend score | clipped [−1, 1] |
| Current allocation | 0.0, 0.5, or 1.0 |
| Days remaining | 1 − step/episode_length |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Buffer size | 50,000 |
| Batch size | 64 |
| Gamma (discount) | 0.99 |
| Target update interval | 1,000 steps |
| Exploration fraction | 0.1 |
| Final epsilon | 0.05 |
| Network architecture | MLP [128, 128] |
| Training timesteps | 200,000 |

---

## 7. Technical Indicators

Computed via the `ta` library on daily OHLCV data:

| Indicator | Parameters | Column |
|-----------|-----------|--------|
| RSI | 14-period | `rsi_14` |
| MACD | 12/26/9 | `macd`, `macd_signal`, `macd_hist` |
| Bollinger Bands | 20-period, 2σ | `bb_upper`, `bb_mid`, `bb_lower`, `bb_pct_b` |
| ATR | 14-period | `atr_14` |
| EMA | 50-period | `ema_50` |
| EMA | 200-period | `ema_200` |
| Trend score | EMA50 vs EMA200 + momentum | `trend_score` (−1 to +1) |
| Daily return | Close pct change | `daily_return` |

---

## 8. Performance Metrics

| Metric | Definition |
|--------|-----------|
| Sharpe ratio | `(mean_daily − rf/252) / std_daily × √252` (rf = 7% India 10Y) |
| Max drawdown | Rolling peak method: `(value − peak) / peak` |
| CAGR | `(end/start)^(1/years) − 1` |
| Win rate | % days with positive portfolio return |
| Constraint: Sharpe | Must be > 1.0 |
| Constraint: MDD | Must be > −10% |

### 2025 Backtest Results (verified)
| Metric | DDQN Strategy | Buy & Hold |
|--------|--------------|------------|
| Total return | +11.08% | +16.65% |
| CAGR | 15.76% | 23.92% |
| Sharpe ratio | 0.951 | — |
| Max drawdown | −4.3% ✓ | — |
| Annual volatility | 8.49% | — |
| Win rate | 38.33% | — |
| Days simulated | 180 | — |

*Sharpe at 0.951 is just below the 1.0 target. MDD constraint (< 10%) is met comfortably.*

---

## 9. Output Files

All outputs written to `rita_output/`:

| File | Contents |
|------|----------|
| `rita_ddqn_model.zip` | Trained Double DQN model weights |
| `session_state.csv` | Flat key-value: goal, strategy, periods, metrics |
| `backtest_daily.csv` | Daily portfolio/benchmark values + allocations |
| `performance_summary.csv` | Aggregated performance metrics |
| `goal_history.csv` | Original + revised goals |
| `monitor_log.csv` | Per-step timing and status |
| `plots/backtest_returns.png` | Cumulative returns vs benchmark |
| `plots/drawdown.png` | Drawdown chart with −10% line |
| `plots/action_distribution.png` | Allocation timeline overlaid on price |
| `plots/rolling_sharpe.png` | 63-day rolling Sharpe ratio |
| `plots/feature_importance.png` | Q-values per feature bucket (interpretability) |

---

## 10. Interfaces

### MCP Server (Claude Desktop)
- 8 MCP tools, one per workflow step
- Entry point: `rita.interfaces.mcp_server:main`
- Config: `config/claude_desktop_config.json` → copy to `%APPDATA%\Claude\`
- Env vars: `NIFTY_CSV_PATH`, `OUTPUT_DIR`, `PYTHONPATH`

### Python Client SDK
```python
from rita.interfaces.python_client import RITAClient

client = RITAClient(r"C:\path\to\merged.csv", output_dir="./rita_output")
client.set_goal(15.0, 365, "moderate")
client.analyze_market()
client.design_strategy()
client.train_model()                     # reuses saved model automatically
client.set_simulation_period()          # defaults to 2025
client.run_backtest()
results = client.get_results()
client.update_goal()
```

### Streamlit Web UI
```bash
python run_ui.py            # auto-selects free port from 8501
python run_ui.py --port 8502
```

**UI features:**
- Sidebar: data source toggle, goal config, force-retrain checkbox, timestep selector, training history options
- 8-step live progress bar during pipeline run; Reset button clears session
- Global KPI strip (8 metrics) always visible above all tabs: Return, CAGR, Sharpe, Max DD, Win Rate, Avg VaR, Train Rounds, Constraints
- **7-tab results dashboard:**

| Tab | Contents |
|-----|---------|
| 🏠 Dashboard | Constraint status badges, goal update summary (Step 8), recommendations |
| 📋 Steps | Interactive 8-step strip — click any step to expand its details inline |
| 📈 Performance | Returns, Drawdown, Rolling Sharpe, Allocations, Q-Value interpretability |
| 🛡️ Risk View | Risk Evolution arc, DD Budget, Trade Impact, Regime & Confidence, Risk-Return Scatter |
| 🔍 Explainability | SHAP Global Importance, Beeswarm, Feature Radar, Dependence Plot, Top Trades |
| 📉 Training | Sharpe/MDD/Return progression across training rounds |
| 📥 Export | JSON summary, HTML report, risk CSVs, training history download |

- Each chart tab shows **4-column card grid**: every card has a 140px mini-preview thumbnail + summary metrics + "🔍 Expand" button that opens the full interactive Plotly chart in a modal dialog
- 📋 Steps tab: 8 bordered cards in one row with `✅/⏳` status; clicking any card expands its step result below (scalars as `st.metric`, nested data in sub-expanders); phase timing table at bottom

### FastAPI REST API (Phase 3)
```bash
python run_api.py                    # port 8000
python run_api.py --port 8080 --reload   # dev mode with auto-reload
```

**Endpoints:**

| Method | Path | Step | Description |
|--------|------|------|-------------|
| POST | `/api/v1/goal` | 1 | Set financial goal |
| POST | `/api/v1/market` | 2 | Analyze market conditions |
| POST | `/api/v1/strategy` | 3 | Design strategy |
| POST | `/api/v1/train` | 4 | Train / load model |
| POST | `/api/v1/period` | 5 | Set simulation period |
| POST | `/api/v1/backtest` | 6 | Run backtest |
| GET | `/api/v1/results` | 7 | Get results + plots |
| POST | `/api/v1/goal/update` | 8 | Update goal |
| POST | `/api/v1/pipeline` | all | Full 8-step run |
| GET | `/health` | — | Service health |
| GET | `/progress` | — | Pipeline progress |
| POST | `/reset` | — | Clear session |

Interactive docs auto-generated at `http://localhost:8000/docs` (Swagger UI).

### Docker
```bash
# Mount your data directory and run:
DATA_DIR=C:\path\to\nifty\data docker compose up api    # REST API
DATA_DIR=C:\path\to\nifty\data docker compose up ui     # Streamlit UI
DATA_DIR=C:\path\to\nifty\data docker compose up        # both
```

---

## 11. Test Suite

39 pytest tests in `tests/test_core.py` — all passing, no CSV or training required for unit tests.

| Class | Tests | Coverage |
|-------|-------|---------|
| `TestPerformanceMetrics` | 9 | Sharpe, MDD, CAGR, compute_all_metrics |
| `TestTechnicalAnalyzer` | 6 | All 13 indicator columns, RSI range, trend validity |
| `TestGoalEngine` | 4 | Feasibility levels, required keys, update logic |
| `TestStrategyEngine` | 4 | Strategy keys, allocation range, constraint pass/fail |
| `TestNiftyTradingEnv` | 6 | Observation shape, action space, step/reset, episode termination |
| `TestAPIEndpoints` | 10 | All 8 workflow endpoints + health + progress (needs CSV) |

```bash
pytest tests/                          # all 39 tests
pytest tests/ -k "not APIEndpoints"   # unit tests only (no CSV needed)
```

---

## 12. Technology Stack

| Category | Library | Version |
|----------|---------|---------|
| RL framework | stable-baselines3 | ≥ 2.3.0 |
| RL environment | gymnasium | ≥ 0.29.0 |
| Deep learning | PyTorch | ≥ 2.0.0 |
| Data processing | pandas | ≥ 2.0.0 |
| Numerical | numpy | ≥ 1.24.0 |
| Technical indicators | ta | ≥ 0.11.0 |
| Static plots | matplotlib | ≥ 3.7.0 |
| Interactive plots | plotly | ≥ 5.0.0 |
| Web UI | streamlit | ≥ 1.35.0 |
| REST API | fastapi | ≥ 0.110.0 |
| API server | uvicorn[standard] | ≥ 0.29.0 |
| API test client | httpx | ≥ 0.27.0 |
| Test framework | pytest | ≥ 7.0.0 |
| MCP protocol | mcp | ≥ 1.0.0, < 2.0.0 |
| Env config | python-dotenv | ≥ 1.0.0 |
| Build system | hatchling | — |
| Container | Docker + docker-compose | — |
| CI/CD | GitHub Actions | — |
| Python | CPython | ≥ 3.10 |

**Python environment:** `C:\Users\Sandeep\pyenv-envs\poc` (shared, not project-level)

---

## 12. Development Phases

### Phase 1 — Core + MCP (Complete ✓)
- All 6 core modules
- Orchestration layer (workflow, session, monitor)
- MCP server (8 tools) + Python client SDK
- End-to-end pipeline verified: 8/8 steps

### Phase 2 — Streamlit UI (Complete ✓)
- Interactive web dashboard
- 5 Plotly charts
- Live step progress
- HTML + JSON export
- Auto port selection

### Phase 3 — REST API + Tests + CI/CD + Docker (Complete ✓)
- FastAPI REST API: 8 endpoints + /pipeline, /health, /progress, /reset
- 39 pytest tests (unit + API integration), all passing
- GitHub Actions CI: runs on every push/PR across Python 3.11 + 3.12
- Dockerfile + docker-compose.yml for containerised deployment

### Phase 4 — UI Enhancements (Complete ✓)
- **Risk Engine** (`risk_engine.py`): risk_timeline, trade_events, risk_summary; VaR, drawdown budget, regime detection
- **Training Tracker** (`training_tracker.py`): appends per-run metrics to `training_history.csv`
- **SHAP Explainability** (`shap_explainer.py`): DeepExplainer on DQN Q-network; global importance, beeswarm, radar, dependence, top trades
- **UI restructure**: 7-tab layout with global KPI strip; chart cards with 140px mini-preview thumbnails + modal expand
- **Interactive Step Strip**: 📋 Steps tab — single-row 8-step status cards, click to expand details inline

---

## 13. Running the Project

### Prerequisites
```powershell
# Activate shared environment
.\activate-env.ps1
pip install -e .
```

### Quick verification (no training, ~10s)
```bash
python verify.py
```

### Full pipeline CLI (~6 min with training)
```bash
python run_pipeline.py
```

### Web UI
```bash
python run_ui.py
```

### REST API
```bash
python run_api.py                       # port 8000
python run_api.py --port 8080 --reload  # dev mode
```
Swagger UI: http://localhost:8000/docs

### Tests
```bash
pytest tests/                           # all 39 tests (needs CSV)
pytest tests/ -k "not APIEndpoints"     # unit tests only (~5s, no CSV)
```

### Claude Desktop MCP
1. Copy `config/claude_desktop_config.json.example` to `config/claude_desktop_config.json`
2. Fill in your local paths
3. Copy to `%APPDATA%\Claude\claude_desktop_config.json`
4. Restart Claude Desktop — RITA tools appear in Claude Cowork

---

## 15. Continuing the Project in a New Session

### Quick orientation (run these first)
```bash
# 1. Activate shared Python env
.\activate-env.ps1

# 2. Confirm everything still works (no training, ~10s)
python verify.py

# 3. Check git status
git log --oneline -5
git status
```

### Context files to read
| File | Purpose |
|------|---------|
| `project-report.md` | Full architecture, design decisions, API reference (this file) |
| `C:\Users\Sandeep\.claude\projects\...\memory\MEMORY.md` | Claude session memory — current status, known bugs, TODOs |
| `pyproject.toml` | All dependencies |
| `src/rita/orchestration/workflow.py` | Central 8-step pipeline — start here to understand flow |

### GitHub repo
```bash
git clone https://github.com/sangaw/riia-cowork-apr-demo.git
cd riia-cowork-apr-demo
.\activate-env.ps1
pip install -e .
```

### Resuming after a gap
The trained model is **not in git** (excluded by `.gitignore`). If `rita_output/rita_ddqn_model.zip`
is missing on your machine, step 4 will retrain automatically (~6 min). To skip retraining, copy
the model zip from a previous run into `rita_output/` before running the pipeline.

### Known pending work
| Item | How to do it |
|------|-------------|
| Push Sharpe above 1.0 | `client.train_model(500_000, force_retrain=True)` |
| Enable Claude Desktop | Copy + edit `config/claude_desktop_config.json.example`, restart Claude |
| Mock data mode | Add synthetic data generator in `data_loader.py`, wire to Streamlit toggle |
| Weekly backtest action | Add `.github/workflows/weekly_backtest.yml` (scheduled cron trigger) |
| PDF report export | Add `reportlab` dep + `generate_pdf_report()` in `performance.py` |

### Architecture reminder — where to add new features
| Feature type | Where to add |
|-------------|-------------|
| New indicator | `src/rita/core/technical_analyzer.py` → `calculate_indicators()` |
| New RL reward component | `src/rita/core/rl_agent.py` → `NiftyTradingEnv.step()` |
| New performance metric | `src/rita/core/performance.py` → `compute_all_metrics()` |
| New workflow step | `src/rita/orchestration/workflow.py` → add `stepN_...()` method |
| New API endpoint | `src/rita/interfaces/rest_api.py` + matching test in `tests/test_core.py` |
| New UI tab/chart | `src/rita/interfaces/streamlit_app.py` → `render_dashboard()` |

---

## 16. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Double DQN over PPO | Discrete action space (3 actions) suits DQN; DDQN reduces overestimation bias |
| 3 actions only (0/50/100%) | Keeps action space small; avoids over-trading; sufficient for index investing |
| No database — CSV only | Simplicity; all state is inspectable text files; no setup required |
| Shared Python env | Avoids project-level venv proliferation across multiple POC projects |
| Drawdown penalty = 2.0 | Original 10.0 overwhelmed the reward signal; 2.0 gives a meaningful but learnable signal |
| Risk-free rate = 7% | India 10Y government bond yield; appropriate for Nifty Sharpe calculation |
| Train/val/backtest fixed splits | Prevents data leakage; 2025 backtest is genuinely out-of-sample |
| Model reuse by default | `step4_train_model(force_retrain=False)` loads existing model; avoids re-running 6-min training on every pipeline call |

---

*Generated: 2026-03-08 · RITA v0.2.0 · GitHub: https://github.com/sangaw/riia-cowork-apr-demo*
