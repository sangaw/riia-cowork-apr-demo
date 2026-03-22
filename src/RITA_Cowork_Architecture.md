# RITA — System Architecture

**v1.0.0 · Updated: March 2026**

---

## Overview

RITA (Risk Informed Trading Approach) is a Nifty 50 RL investment system with a strict 3-layer architecture. It exposes four interfaces — Claude Desktop MCP, HTML dashboard, Streamlit, and REST API — all routing through the same orchestration and core layers.

---

## Layer Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INTERFACE LAYER                                │
│                                                                          │
│  ┌───────────────┐  ┌──────────────────┐  ┌───────────┐  ┌──────────┐  │
│  │ Claude Desktop│  │  HTML Dashboard  │  │ Streamlit │  │ REST API │  │
│  │  MCP (14 tools│  │  :8000/dashboard/│  │   :8501   │  │  :8000   │  │
│  │  mcp_server.py│  │  rita.html +     │  │ 11 tabs   │  │ 24+ endp │  │
│  │               │  │  fno.html +      │  │           │  │ FastAPI  │  │
│  │               │  │  ops.html        │  │           │  │          │  │
│  └───────┬───────┘  └────────┬─────────┘  └─────┬─────┘  └────┬─────┘  │
└──────────┼───────────────────┼─────────────────┼─────────────┼─────────┘
           │                   │                 │             │
           └───────────────────┴────────┬────────┘             │
                                        │                      │
┌───────────────────────────────────────▼──────────────────────▼──────────┐
│                         ORCHESTRATION LAYER                              │
│                                                                          │
│   workflow.py         — WorkflowOrchestrator (8-step pipeline runner)   │
│   session.py          — SessionManager (CSV state persistence)           │
│   monitor.py          — PhaseMonitor (step timing + logging)             │
│                                                                          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│                              CORE LAYER                                  │
│                                                                          │
│  data_loader.py       — CSV load, train/val/test splits, bear episodes  │
│  technical_analyzer.py— RSI, MACD, BB, ATR, EMA, regime detection      │
│  rl_agent.py          — NiftyTradingEnv, DDQN training, backtest        │
│  performance.py       — Sharpe, MDD, CAGR, portfolio comparison         │
│  risk_engine.py       — VaR timeline, trade events, drawdown budget     │
│  shap_explainer.py    — SHAP DeepExplainer on DQN Q-network             │
│  training_tracker.py  — Per-round metrics → training_history.csv        │
│  goal_engine.py       — Goal setting, feasibility, feedback             │
│  strategy_engine.py   — HOLD/HALF/FULL proxy, constraint validation     │
│  classifier.py        — Chat intent classification (20 intents, 6 hdlrs)│
│  chat_monitor.py      — CSV-based chat query logging                    │
│                                                                          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│                              DATA LAYER                                  │
│                                                                          │
│  Local CSV only — no external APIs                                       │
│  NIFTY_CSV_PATH     — base Nifty 50 OHLCV CSV                           │
│  rita_input/        — drop NSE-format CSVs here for ingestion           │
│  rita_output/       — model weights, output CSVs, plots                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## RL Model Flow

```
Historical OHLCV (2010-2022)
         │
         ▼
  technical_analyzer.py
  calculate_indicators()
  → 9-feature state vector
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
 Bull Training               Bear Training
 500k steps, n_seeds=5       200k steps, max 3 seeds
 Full training set           Correction episodes only
 get_bear_episodes() →      (~600 rows, 2010-2022)
    └────┬────────────────────────┘
         │
         ▼
  Regime Detection
  detect_regime(df)
  ema_26/ema_50 < 0.99 for 3+ days?
         │
    BEAR ├──────────────────────► Bear model
    BULL └──────────────────────► Bull model
         │
         ▼
  run_regime_episode()
  → backtest_daily.csv
  → performance_summary.csv
```

---

## Chat Flow (HTML Dashboard)

```
User types question in Market Analysis page
         │
         ▼
  POST /api/v1/chat
         │
         ▼
  classifier.py
  classify(query)
  → cosine similarity vs 140 seed phrases
  → IntentResult{intent, confidence, low_confidence}
         │
         ▼
  dispatch(result, df, portfolio_inr, output_dir)
  → calls Python handler function
  → fills response template
         │
         ▼
  chat_monitor.py → log_query() → chat_monitor.csv
         │
         ▼
  {intent, handler, confidence, low_confidence, response, latency_ms}
         │
         ▼
  rendered in Panel 2 chat bubble
```

---

## 8-Step Pipeline (workflow.py)

```
step1_set_financial_goal(target_pct, horizon_days, risk_tolerance)
    → session_state.csv: goal fields

step2_analyze_market()
    → technical_analyzer.calculate_indicators()
    → session_state.csv: market fields

step3_design_strategy()
    → strategy_engine.get_strategy_recommendation()
    → session_state.csv: strategy fields

step4_train_model(model_type="bull"|"bear"|"both", timesteps, force_retrain, n_seeds)
    → rl_agent.train_best_of_n() or train_bear_model()
    → rita_output/rita_ddqn_model.zip (bull)
    → rita_output/rita_ddqn_bear_model.zip (bear)
    → training_history.csv

step5_set_simulation_period(start_date, end_date)
    → session_state.csv: period fields

step6_run_backtest(backtest_mode="auto"|"bull"|"regime")
    → rl_agent.run_regime_episode() or run_episode()
    → backtest_daily.csv, backtest_trades.csv

step7_get_results()
    → performance.compute_all_metrics()
    → shap_explainer.run_shap_analysis()
    → risk_engine.build_risk_timeline()
    → performance_summary.csv, risk_timeline.csv, shap_importance.csv
    → plots/

step8_update_goal()
    → goal_engine.update_goal()
    → goal_history.csv
```

---

## MCP Tool Routing

```
Claude Desktop
    │ stdio
    ▼
mcp_server.py
    │
    ├── Standalone tools (no session required) ─────────────────────────┐
    │   get_return_estimates()          → data_loader.get_historical_stats()
    │   get_market_sentiment()          → technical_analyzer.get_market_sentiment()
    │   get_strategy_recommendation()   → strategy_engine.get_strategy_recommendation()
    │   get_portfolio_scenarios()       → performance.build_portfolio_comparison()
    │   get_stress_scenarios()          → performance.simulate_stress_scenarios()
    │   get_performance_feedback()      → performance.get_performance_feedback()
    │                                                                    │
    └── Pipeline tools (session-bound, sequential) ──────────────────────┘
        step1 → step8 → WorkflowOrchestrator → Core layer functions
```

---

## HTML Dashboard Navigation

```
dashboard/index.html   (landing, 4 cards)
    │
    ├── dashboard/rita.html  (main RITA app)
    │       Phase 01 — Plan (green):
    │         Data Prep · Financial Goal · Market Analysis · Market Signals · Strategy
    │       Phase 02 — Build (orange):
    │         Train Model
    │       Phase 03 — Analyse (blue):
    │         Performance · Trade Journal · Explainability
    │       Phase 04 — Monitor (purple):
    │         Risk View · Training Progress · Observability · MCP Calls · Audit
    │
    ├── dashboard/fno.html   (FnO Portfolio Manager, 5 pages)
    │       Dashboard · Positions · Margin Tracker · Risk & Greeks · Risk-Reward
    │       BANKNIFTY scope only — never mixed into RL pipeline
    │
    └── dashboard/ops.html   (Operations Portal, 6 sections)
            Overview · Monitoring · CI/CD · Deployments · Observability · Chat Analytics
```

---

## Technology Stack

| Category | Library |
|----------|---------|
| RL framework | stable-baselines3 (DQN) |
| RL environment | gymnasium |
| Deep learning | PyTorch |
| Data processing | pandas, numpy |
| Technical indicators | ta |
| Chat classification | sentence-transformers (all-MiniLM-L6-v2) |
| REST API | FastAPI + uvicorn |
| Web UI | streamlit |
| Charts | plotly (Streamlit), Chart.js 4.4 (HTML) |
| Static plots | matplotlib |
| Tests | pytest |
| Build | hatchling |
| CI/CD | GitHub Actions |
| Container | Docker + docker-compose |
| Python | ≥ 3.10 (tested on 3.11, 3.12) |
