# RITA — Project Report
**Risk Informed Trading Approach · Nifty 50 RL Investment System**

---

## 1. Project Overview

RITA is a Nifty 50 index investment system powered by a Reinforcement Learning (Double DQN) model. It follows an 8-step workflow — from goal setting and market analysis, through model training and backtesting, to goal feedback — and is callable through three interfaces: MCP (Claude Desktop), a Python client SDK, and a Streamlit web UI.

### Core idea
Given only historical Nifty 50 OHLCV data, train an RL agent to decide daily: hold cash (0%), go half-invested (50%), or fully invested (100%). Constrain the strategy to achieve Sharpe ratio > 1.0 and maximum drawdown < 10%.

---

## 2. Architecture

RITA follows a strict 3-layer architecture. No layer accesses a lower layer's internals — all communication goes through defined function calls.

```
┌──────────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                          │
│  MCP Server (Claude Desktop)  |  Streamlit UI  |  Python SDK │
└────────────────────┬─────────────────────────────────────────┘
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
│       └── streamlit_app.py        # Web UI (Phase 2)
├── run_pipeline.py                 # CLI: full 8-step pipeline
├── run_ui.py                       # CLI: launch Streamlit UI
├── verify.py                       # Quick test: steps 1–3, no training
├── test_steps5to8.py               # Quick test: steps 5–8 with saved model
├── pyproject.toml                  # Package + dependencies
├── activate-env.ps1                # Activate shared Python env
└── config/
    └── claude_desktop_config.json  # MCP config for Claude Desktop
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
streamlit run src/rita/interfaces/streamlit_app.py
```

**UI features:**
- Sidebar: data source toggle, goal config, model toggle, sim period
- Step-by-step live progress with `st.status` blocks
- Results dashboard: Overview · Charts · Step Details · Export
- 5 interactive Plotly charts
- JSON + HTML report download

---

## 11. Technology Stack

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
| MCP protocol | mcp | ≥ 1.0.0, < 2.0.0 |
| Env config | python-dotenv | ≥ 1.0.0 |
| Build system | hatchling | — |
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

### Phase 3 — REST API + CI/CD (Planned)
- FastAPI exposing all 8 workflow steps as REST endpoints
- GitHub Actions: CI on push, weekly automated backtest
- Docker containerisation

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

### Claude Desktop MCP
1. Copy `config/claude_desktop_config.json` content to `%APPDATA%\Claude\claude_desktop_config.json`
2. Restart Claude Desktop
3. RITA tools appear in Claude Cowork

---

## 14. Key Design Decisions

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

*Generated: 2026-03-05 · RITA v0.2.0*
