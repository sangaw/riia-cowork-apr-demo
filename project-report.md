# RITA — Project Report
**Risk Informed Trading Approach · Nifty 50 RL Investment System · v1.0.0**
*Generated: 2026-03-22*

---

## 1. Project Overview

RITA is a Nifty 50 index investment system powered by a Dual-Model Reinforcement Learning (Double DQN) engine. An 8-step workflow moves from goal setting through model training and backtesting to goal feedback. The system is callable through four interfaces: Claude Desktop MCP (14 tools), a Python client SDK, a Streamlit web UI, and a FastAPI REST API. A separate HTML/JS business dashboard provides a polished front end for non-technical users.

**Core idea:** Given only historical Nifty 50 OHLCV data, train an RL agent to decide daily: hold cash (0%), go half-invested (50%), or fully invested (100%). Use regime detection to route between a Bull model (growth focus) and a Bear model (capital preservation). Constrain the strategy: Sharpe ratio > 1.0 AND maximum drawdown < 10%.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                             │
│  Claude Desktop MCP (14 tools) │ HTML Dashboard │ Streamlit │ REST  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                            │
│      WorkflowOrchestrator · SessionManager · PhaseMonitor          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                          CORE LAYER                                 │
│  data_loader · technical_analyzer · rl_agent · performance         │
│  goal_engine · strategy_engine · classifier · chat_monitor         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
rita-cowork-demo/
├── src/rita/
│   ├── core/
│   │   ├── data_loader.py          # CSV load, splits, bear episode extraction
│   │   ├── technical_analyzer.py   # RSI, MACD, BB, ATR, EMA, trend score, regime detection
│   │   ├── rl_agent.py             # NiftyTradingEnv + Double DQN, train_best_of_n, run_regime_episode
│   │   ├── performance.py          # Sharpe, MDD, CAGR, portfolio comparison, stress test
│   │   ├── risk_engine.py          # VaR timeline, trade events, drawdown budget
│   │   ├── shap_explainer.py       # SHAP DeepExplainer on DQN Q-network
│   │   ├── training_tracker.py     # Per-round training history (CSV)
│   │   ├── goal_engine.py          # Goal setting + feasibility + feedback
│   │   ├── strategy_engine.py      # Constraint validation, HOLD/HALF/FULL proxy
│   │   ├── classifier.py           # Chat intent classifier (sentence-transformers, 20 intents)
│   │   └── chat_monitor.py         # CSV-based chat query logging (no DB)
│   ├── orchestration/
│   │   ├── workflow.py             # WorkflowOrchestrator (8-step runner)
│   │   ├── session.py              # SessionManager (CSV state persistence)
│   │   └── monitor.py              # PhaseMonitor (timing + step logging)
│   └── interfaces/
│       ├── mcp_server.py           # MCP stdio server (14 tools)
│       ├── python_client.py        # RITAClient Python SDK
│       ├── streamlit_app.py        # Streamlit web UI (11 tabs)
│       └── rest_api.py             # FastAPI (24+ endpoints + static HTML dashboard)
├── dashboard/
│   ├── index.html                  # Landing page (4 cards)
│   ├── rita.html                   # Main RITA app (~2500 lines, vanilla JS + Chart.js)
│   ├── fno.html                    # FnO Portfolio Manager (5 pages)
│   └── ops.html                    # Operations Portal (6 sections)
├── tests/
│   └── test_core.py                # 39 pytest tests (29 unit + 10 API integration)
├── .github/workflows/ci.yml        # GitHub Actions (Python 3.11 + 3.12)
├── Dockerfile + docker-compose.yml
├── run_pipeline.py / run_ui.py / run_api.py / verify.py
└── config/claude_desktop_config.json.example
```

---

## 4. The 8-Step Workflow

| Step | Method | What it does |
|------|--------|-------------|
| 1 | `step1_set_financial_goal` | Set target return %, horizon, risk tolerance. Validate against Nifty history. |
| 2 | `step2_analyze_market` | Compute RSI, MACD, BB, ATR, EMA on last 252 days. Classify trend + sentiment. |
| 3 | `step3_design_strategy` | Pick allocation approach (HOLD/HALF/FULL) from trend × risk matrix. |
| 4 | `step4_train_model(model_type)` | Train bull / bear / both. bull=500k/5-seeds; bear=200k/3-seeds on correction episodes. |
| 5 | `step5_set_simulation_period` | Set backtest window (default: 2025-01-01 → latest in CSV). |
| 6 | `step6_run_backtest(backtest_mode)` | Run model (auto / bull / regime). Compute performance metrics. |
| 7 | `step7_get_results` | Generate charts. Run constraint check (Sharpe > 1.0, MDD < 10%). Return report. |
| 8 | `step8_update_goal` | Compare actual vs target. Revise goal for next cycle. |

---

## 5. Data

| Property | Detail |
|----------|--------|
| Source | Local CSV — no external API calls |
| Instrument | Nifty 50 index (daily OHLCV) |
| Coverage | 1999-07-01 → 20-Mar-2026 (6,650 rows) |
| Columns | `date, open, high, low, close, shares traded, turnover (₹ cr)` |
| Training | 2010-01-01 → 2022-12-31 |
| Validation | 2023-01-01 → 2024-12-31 |
| Backtest | 2025-01-01 → latest |
| Input dir | `rita_input/` — drop NSE-format CSVs; run Data Prep to merge |

**Warning:** `banknifty_manual.csv` must not be in `rita_input/` when running Data Prep — it would be merged into the NIFTY series.

---

## 6. RL Model — v1.5 Dual Model + Regime Routing

### Algorithm: Double DQN
Implemented via stable-baselines3 DQN. Soft target updates (`tau=0.005, target_update_interval=1`), Monitor wrapper for `ep_rew_mean` logging, `exploration_fraction=0.5`, network [256, 256].

### State Space (9 features)

| Feature | Normalisation |
|---------|--------------|
| Daily return | × 10, clipped ±3 |
| RSI-14 | ÷ 100 → [0, 1] |
| MACD | z-score ÷ (3 × std), clipped ±3 |
| Bollinger %B | clipped [−0.5, 1.5] |
| Trend score | clipped [−1, 1] |
| Current allocation | 0.0, 0.5, or 1.0 |
| Days remaining | 1 − step/episode_length |
| ATR norm | atr_14 / atr_mean, clipped [0, 3] |
| EMA ratio norm | clip((ema_26/ema_50 − 1) × 20, −3, 3) |

### Bull Model
- Training: 500k timesteps, n_seeds=5 (best-of-N), full 2010–2022 set
- Reward: `portfolio_ret − 0.005` flat penalty if cumulative drawdown < −10%
- Purpose: growth optimisation in normal market conditions

### Bear Model
- Training: 200k timesteps, max 3 seeds, correction episodes only (~600 rows)
- Correction episodes extracted by `get_bear_episodes()` — min 20 trading days, 10-day buffer
- Reward: `portfolio_ret − max(0, (|drawdown| − 0.03) × 1.0)` — proportional penalty beyond −3%
- Purpose: capital preservation; mostly-cash is optimal policy

### Regime Detection
```python
detect_regime(df, consecutive_days=3)
# Returns BEAR if ema_26/ema_50 < 0.99 for consecutive_days+ days; else BULL
```

### Regime-Aware Backtest
```python
run_regime_episode(bull_model, test_df, bear_model, consecutive_bear_days=3)
# Switches model day-by-day based on rolling bear count
# Falls back to bull-only if bear_model=None or ema_ratio column missing
```

### Why Feature 9 Was Added
The 8-feature model showed Sharpe 1.253 through Dec 2025, falling to 0.826 by Feb 2026. Root cause: `trend_score` (EMA-50 slope) stayed positive throughout the Jan–Feb 2026 correction, causing 31 extra trades ("dead-cat bounce churning"). `ema_ratio_norm` turns negative within 2–3 weeks of a trend break, enabling earlier detection.

---

## 7. Technical Indicators

Computed via the `ta` library on daily OHLCV data:

| Indicator | Parameters | Column(s) |
|-----------|-----------|----------|
| RSI | 14-period | `rsi_14` |
| MACD | 12/26/9 | `macd, macd_signal, macd_hist` |
| Bollinger Bands | 20-period, 2σ | `bb_upper, bb_mid, bb_lower, bb_pct_b` |
| ATR | 14-period | `atr_14` |
| EMA | 5, 13, 26, 50, 200-period | `ema_5, ema_13, ema_26, ema_50, ema_200` |
| Trend score | EMA-50 vs EMA-200 + momentum | `trend_score` (−1 to +1) |
| Daily return | Close pct change | `daily_return` |

---

## 8. Performance Metrics

| Metric | Definition |
|--------|-----------|
| Sharpe ratio | `(mean_daily − rf/252) / std_daily × √252` (rf = 7% India 10Y) |
| Max drawdown | Rolling peak method: `(value − peak) / peak` |
| CAGR | `(end/start)^(1/years) − 1` |
| Win rate | % days with positive portfolio return |

### Current Backtest (Apr 2025 – Feb 2026, 221 days)
| Metric | RITA (8-feature) | Buy & Hold |
|--------|-----------------|-----------|
| Return | 12.0% | 12.41% |
| Sharpe | 0.826 | — |
| Max drawdown | −3.49% | — |
| Trades | 111 | — |
| Win rate | 32.58% | — |

*9-feature bull + bear models not yet trained. Retrain expected to improve Sharpe.*

---

## 9. Output Files

All written to `rita_output/`:

| File | Contents |
|------|----------|
| `rita_ddqn_model.zip` | Bull model weights |
| `rita_ddqn_bear_model.zip` | Bear model weights |
| `session_state.csv` | Flat key-value: goal, strategy, periods, metrics |
| `backtest_daily.csv` | Daily portfolio/benchmark values + allocations |
| `performance_summary.csv` | Aggregated performance metrics |
| `backtest_trades.csv` | Individual trade log |
| `training_history.csv` | Per-round training metrics |
| `mcp_call_log.csv` | All MCP tool calls with timing |
| `chat_monitor.csv` | Chat query log (intent, handler, confidence, latency) |
| `monitor_log.csv` | Per-step pipeline timing |
| `shap_importance.csv` | SHAP feature importances |
| `risk_timeline.csv` | VaR and drawdown timeline |

---

## 10. REST API Endpoints

All served by `python run_api.py` on port 8000:

### Pipeline Endpoints
| Method | Path | Step | Description |
|--------|------|------|-------------|
| POST | `/api/v1/goal` | 1 | Set financial goal |
| POST | `/api/v1/market` | 2 | Analyze market |
| POST | `/api/v1/strategy` | 3 | Design strategy |
| POST | `/api/v1/train` | 4 | Train / load model |
| POST | `/api/v1/period` | 5 | Set simulation period |
| POST | `/api/v1/backtest` | 6 | Run backtest |
| GET | `/api/v1/results` | 7 | Get results |
| POST | `/api/v1/goal/update` | 8 | Update goal |
| POST | `/api/v1/pipeline` | all | Full 8-step run |
| GET | `/health` | — | Service health |
| GET | `/progress` | — | Pipeline progress |
| POST | `/reset` | — | Clear session |

### Data / Dashboard Endpoints
| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/v1/backtest-daily` | Daily portfolio + benchmark + allocation |
| GET | `/api/v1/performance-summary` | Aggregated metrics dict |
| GET | `/api/v1/training-history` | Per-round training CSV |
| GET | `/api/v1/market-signals?timeframe=daily&periods=252` | OHLCV + all indicators (live, mtime-cached) |
| GET | `/api/v1/shap` | SHAP feature importances |
| GET | `/api/v1/risk-timeline` | VaR + drawdown timeline |
| GET | `/api/v1/step-log` | Per-step timing |
| GET | `/api/v1/mcp-calls?limit=100` | MCP call log (newest first) |
| GET | `/api/v1/data-prep/status` | Active CSV info + rita_input/ file list |
| POST | `/api/v1/data-prep/run` | Merge rita_input/ CSVs → nifty_merged.csv |
| POST | `/api/v1/chat` | Intent classification + response |
| GET | `/api/v1/chat/monitor` | Chat summary, recent queries, intent distribution |
| GET | `/metrics` | Observability metrics |
| GET | `/api/v1/drift` | Drift detection results |

Static mount: `app.mount("/dashboard", StaticFiles(directory="dashboard", html=True))`

---

## 11. MCP Tools (14)

### Standalone Conversational Tools (load CSV directly, no session needed)
| Tool | Trigger | What it returns |
|------|---------|----------------|
| `get_return_estimates` | "What returns can I expect?" | 5 percentile scenarios, win rate, suggested target |
| `get_market_sentiment` | "How is the market?" | BULLISH/NEUTRAL/BEARISH, 5-signal score −6 to +6 |
| `get_strategy_recommendation` | "What allocation?" | HOLD/HALF/FULL, rationale, override rules |
| `get_portfolio_scenarios` | "Show me scenarios for ₹X" | Conservative/Moderate/Aggressive vs RITA in INR |
| `get_stress_scenarios` | "What if market moves ±20%?" | Stress test across all profiles |
| `get_performance_feedback` | "How did RITA perform?" | Return, CAGR, Sharpe, MDD, trades, expectations |

### 8-Step Pipeline Tools
`step1_set_financial_goal` → `step2_analyze_market` → `step3_design_strategy` → `step4_train_model` → `step5_set_simulation_period` → `step6_run_backtest` → `step7_get_results` → `step8_update_goal`

All 14 tools log to `rita_output/mcp_call_log.csv` after every call.

---

## 12. Chat Classifier

Embedded in the HTML dashboard Market Analysis page. No Claude API at runtime.

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers) — loads once as singleton
- **Index:** ~140 seed phrases encoded to (140×384) numpy matrix — O(140) cosine dot product
- **20 intents** across 6 handlers:

| Handler | Python function | Intent examples |
|---------|----------------|-----------------|
| `market_sentiment` | `get_market_sentiment(df)` | trend, RSI, volatility, overbought |
| `strategy_recommendation` | `get_strategy_recommendation(df)` | invest now, allocation, conservative/aggressive |
| `return_estimates` | `get_period_return_estimates(df, period)` | 1m/3m/6m/1y/3y/5y returns |
| `stress_scenarios` | `simulate_stress_scenarios(df, portfolio_inr)` | Nifty −10%, −20%, +10%, flat |
| `performance_feedback` | `get_performance_feedback(perf_summary)` | how RITA performed, backtest stats |
| `portfolio_comparison` | `build_portfolio_comparison(backtest_daily)` | compare profiles vs RITA model |

- **CONFIDENCE_THRESHOLD = 0.42** — below this the response flags low confidence
- REST endpoint: `POST /api/v1/chat` → `{intent, handler, confidence, low_confidence, response, latency_ms}`

---

## 13. HTML Dashboard

Vanilla JS + Chart.js 4.4 + `chartjs-plugin-annotation@3.0.1`. Design system: Instrument Serif + Epilogue + IBM Plex Mono.

### Files
| File | Purpose |
|------|---------|
| `dashboard/index.html` | Landing page — 4 cards (`repeat(4,1fr)`, max-width 1380px) |
| `dashboard/rita.html` | Main RITA app (~2500 lines) |
| `dashboard/fno.html` | FnO Portfolio Manager (5 pages) |
| `dashboard/ops.html` | Operations Portal (6 sections) |

### Navigation (rita.html)
- **Phase 01 — Plan (green):** Data Prep, Financial Goal, Market Analysis, Market Signals, Strategy
- **Phase 02 — Build (orange):** Train Model
- **Phase 03 — Analyse (blue):** Performance, Trade Journal, Explainability
- **Phase 04 — Monitor (purple):** Risk View, Training Progress, Observability, MCP Calls, Audit

### Market Analysis — 3-Panel Layout
Grid: `5fr 5fr 3fr`, fixed height `calc(100vh - 52px - 116px)`
- Panel 1: Current Market Analysis + "Analyze Market" button
- Panel 2: Chat bubbles + session stats + Clear button; POST /api/v1/chat
- Panel 3: 10 chip buttons (auto-send on click)

### Market Signals Page
- Daily / Weekly / Monthly tab switcher
- Data range label (dates + bar count)
- Signal KPI strip: RSI-14, MACD, BB %B, EMA 5, EMA 13
- Alerts strip: RSI overbought/oversold, MACD direction, BB band breaks, EMA crossovers
- Charts (all expand on click): Price+Volume, RSI, MACD, Bollinger Bands, EMA Overlays
- EMA crossover triangles: EMA 5 vs EMA 13 (green ▲ / red ▼)

### Design System
```css
--bg:#F5F3EE  --surface:#FFFFFF  --surface2:#F9F8F5
--build:#1A6B3C  --run:#0056B8  --mon:#6B2FA0
--warn:#92480A  --danger:#9B1C1C
```

### Critical Chart.js Pattern
```html
<div class="chart-box" style="height:220px"><canvas id="chart-xxx"></canvas></div>
```
Never set height on `<canvas>` directly when `maintainAspectRatio: false`.

---

## 14. Streamlit UI

11 tabs after pipeline run:

| Tab | Contents |
|-----|---------|
| 🏠 Dashboard | KPI strip (8 metrics), constraint badges, goal summary |
| 📋 Steps | Interactive 8-step strip with expandable detail panels |
| 📈 Performance | Returns, drawdown, Sharpe, Q-values |
| 🛡️ Risk View | Risk timeline, VaR, trade impact, regime breakdown |
| 📓 Trade Journal | Trade log, win/loss breakdown |
| 🔍 Explainability | SHAP global importance, beeswarm, radar, dependence |
| 📉 Training | Round history, Sharpe/MDD progression |
| 📥 Export | JSON, HTML, CSV downloads |
| 🔭 Observability | Drift detection, API latency, health checks |
| 🚀 DevOps | Docker, CI/CD status |
| 🔌 MCP Calls | Isolated refresh (🔄), call log, per-tool latency |

---

## 15. Test Suite

39 pytest tests in `tests/test_core.py` — all passing.

```bash
pytest tests/                          # all 39 (requires NIFTY_CSV_PATH)
pytest tests/ -k "not APIEndpoints"   # 29 unit tests (~5s, no CSV)
```

| Class | Tests | Coverage |
|-------|-------|---------|
| `TestPerformanceMetrics` | 9 | Sharpe, MDD, CAGR, compute_all_metrics |
| `TestTechnicalAnalyzer` | 6 | All indicator columns, RSI range, trend validity |
| `TestGoalEngine` | 4 | Feasibility levels, required keys, update logic |
| `TestStrategyEngine` | 4 | Strategy keys, allocation range, constraint pass/fail |
| `TestNiftyTradingEnv` | 6 | Observation shape, action space, step/reset, episode end |
| `TestAPIEndpoints` | 10 | All 8 workflow endpoints + health + progress |

---

## 16. Running the Project

```powershell
# Activate shared env
. .\activate-env.ps1

# Quick check (no training, ~10s)
python verify.py

# Full pipeline (reuses saved model, ~30s)
python run_pipeline.py

# HTML dashboard + API
python run_api.py          # port 8000

# Streamlit UI
python run_ui.py           # port 8501

# Tests
pytest tests/ -k "not APIEndpoints"   # fast, no CSV
pytest tests/                          # full suite

# Claude Desktop MCP
# 1. Copy config/claude_desktop_config.json.example → fill in paths
# 2. Copy to %APPDATA%\Claude\claude_desktop_config.json
# 3. Restart Claude Desktop
```

---

## 17. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Double DQN | Discrete action space (3 actions); DDQN reduces overestimation bias |
| 3 actions only (0/50/100%) | Small space; avoids over-trading; sufficient for index investing |
| Dual model + regime routing | Single model cannot handle both bull and bear regimes equally well |
| Feature 9 (ema_ratio) | `trend_score` too slow to detect corrections; ema_ratio responds within 2–3 weeks |
| No database — CSV only | POC simplicity; state is inspectable plain text; no infrastructure required |
| Shared Python env | Avoids project-level venv proliferation across POC projects |
| Best-of-N seeds | DDQN is sensitive to random seeds; multi-seed guards against lucky/unlucky initialisation |
| Standalone MCP tools | Tools 1–6 work without a live session — faster, more robust for conversational use |
| Chat via cosine similarity | No Claude API at runtime — deterministic, fast, no cost, fully testable |
| Risk-free rate = 7% | India 10Y government bond yield; appropriate for Nifty Sharpe calculation |

---

## 18. Continuing After a Gap

```powershell
# 1. Activate env
. .\activate-env.ps1

# 2. Quick check
python verify.py

# 3. Orient
git log --oneline -5
git status
```

Model files (`rita_output/*.zip`) are excluded from git. Copy from a previous run or retrain:
- In Streamlit: **🏗 Build Model Pipeline** (force_retrain=True)
- Or: `python run_pipeline.py` with `force_retrain=True` in the script

### Where to Add New Features
| Feature type | Where |
|-------------|-------|
| New indicator | `technical_analyzer.py` → `calculate_indicators()` |
| New RL reward component | `rl_agent.py` → `NiftyTradingEnv.step()` |
| New performance metric | `performance.py` → `compute_all_metrics()` |
| New workflow step | `workflow.py` → add `stepN_...()` method |
| New API endpoint | `rest_api.py` + test in `tests/test_core.py` |
| New Streamlit tab | `streamlit_app.py` → `render_dashboard()` |
| New HTML dashboard page | `dashboard/rita.html` → add nav item + section |
| New chat intent | `classifier.py` → extend `INTENT_SEEDS` |
