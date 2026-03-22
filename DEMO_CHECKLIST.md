# RITA Demo Checklist

**Version:** 1.0.0 · **Status:** Ready for Demo · **Last Updated:** March 2026

---

## Pre-Demo Setup

```powershell
cd C:\path\to\rita-cowork-demo
. .\activate-env.ps1

# Start both servers
python run_api.py    # Terminal 1 — HTML dashboard + FastAPI
python run_ui.py     # Terminal 2 — Streamlit
```

Verify:
- [ ] http://localhost:8000/dashboard/ loads (4 cards)
- [ ] http://localhost:8501 loads (Streamlit)
- [ ] Claude Desktop shows RITA tools (rita-cowork MCP server connected)

Run pipeline to ensure output CSVs exist:
- [ ] In Streamlit → click **▶ Re-use Model Pipeline**
- [ ] Or: `python run_pipeline.py`

---

## What's Implemented

### RL Model (v1.5 — Dual Model + Regime Routing)
- [x] Double DQN trained on Nifty 50 (2010–2022)
- [x] **9-feature** state space: daily_return, RSI-14, MACD, BB %B, trend_score, allocation, days_remaining, ATR_norm, **ema_ratio_norm** (regime signal)
- [x] 3-action space: HOLD (0%), HALF (50%), FULL (100%)
- [x] Bull model: 500k timesteps, best-of-N seeds training
- [x] Bear model: correction episodes, 200k timesteps, capital preservation
- [x] Regime detection: `ema_26/ema_50 < 0.99` for 3+ days → BEAR
- [x] Regime-aware backtest: switches bull/bear model daily

### MCP Integration (14 tools)
- [x] `get_return_estimates` — historical Nifty scenarios by period
- [x] `get_market_sentiment` — BULLISH/NEUTRAL/BEARISH from 5 signals (score −6 to +6)
- [x] `get_strategy_recommendation` — HOLD/HALF/FULL, mirrors RL action space
- [x] `get_portfolio_scenarios` — Conservative/Moderate/Aggressive vs RITA in INR
- [x] `get_stress_scenarios` — ±market stress test across all profiles
- [x] `get_performance_feedback` — full backtest summary + realistic expectations
- [x] `step1` → `step8` — full 8-step pipeline via MCP

### HTML Dashboard (:8000)
- [x] Landing page: 4 cards — RIIA App, Data Scientist App, Portfolio Manager, Operations Portal
- [x] **Phase 01 — Plan:** Data Prep, Financial Goal, Market Analysis (with RITA chat), Market Signals, Strategy
- [x] **Phase 02 — Build:** Train Model (bull/bear/both + regime backtest modes)
- [x] **Phase 03 — Analyse:** Performance, Trade Journal, Explainability
- [x] **Phase 04 — Monitor:** Risk View, Training Progress, Observability, MCP Calls, Audit
- [x] Market Analysis 3-panel chat: intent-based (no Claude API — deterministic OHLCV responses)
- [x] FnO Portfolio Manager: 5 pages (Dashboard, Positions, Margin, Risk & Greeks, Risk-Reward)
- [x] Operations Portal: 6 sections incl. Chat Analytics (with Python function mapping)

### Streamlit UI (:8501)
- [x] 11 tabs: Dashboard, Steps, Performance, Risk View, Trade Journal, Explainability, Training, Export, Observability, DevOps, MCP Calls
- [x] Build Model Pipeline (force_retrain=True) + Re-use Model Pipeline buttons
- [x] Model type radio: bull / bear / both
- [x] Backtest mode radio: auto / bull / regime
- [x] Training: Timesteps slider, Seeds slider

### Infrastructure
- [x] FastAPI REST API (24+ endpoints) with pytest suite (39 tests, all passing)
- [x] Docker + GitHub CI/CD (Python 3.11 + 3.12)
- [x] SHAP explainability (DeepExplainer on DQN Q-network)
- [x] Risk engine (VaR timeline, trade events, phase breakdown)
- [x] Training tracker (round history CSV)
- [x] Chat classifier (sentence-transformers, 20 intents, 6 handlers)
- [x] Chat monitor (CSV-based query logging, ops analytics)

---

## Demo Flow — Claude Desktop

Start a new conversation. RITA tools load automatically.

### Step 1 — Return Expectations
> "What returns can I expect from Nifty 50 over 1 year?"

→ `get_return_estimates`: 5 percentile scenarios (conservative to best-case), win rate, suggested target.

### Step 2 — Market Sentiment
> "How is the market looking right now?"

→ `get_market_sentiment`: BULLISH/NEUTRAL/BEARISH from EMA cross, MACD, RSI, Bollinger, ATR.

### Step 3 — Strategy Recommendation
> "What allocation should I take in Nifty 50?"

→ `get_strategy_recommendation`: HOLD / HALF / FULL. Rationale, override rules, upgrade/downgrade triggers.

### Step 4 — Portfolio Scenarios
> "I have 10 lakh INR. Show me how different strategies would perform."

→ `get_portfolio_scenarios`: Conservative/Moderate/Aggressive vs RITA — final INR values, Sharpe, MDD, return.

### Step 5 — Stress Test
> "What if the market moves 20% up or down from here?"

→ `get_stress_scenarios`: Stress across all profiles + RITA current + RITA→HOLD. Shows ₹ P&L, drawdown breach flag.

### Step 6 — Performance Feedback
> "How did RITA perform? What are realistic expectations going forward?"

→ `get_performance_feedback`: Return %, CAGR, Sharpe, MDD, trade count, time at each allocation, constraint verdict.

---

## Demo Flow — HTML Dashboard

1. Open http://localhost:8000/dashboard/ → click **RIIA App**
2. **Phase 01 → Market Analysis**: Click "Analyze Market", then type a question in the chat or click a suggestion chip
3. **Phase 01 → Market Signals**: Show RSI, MACD, BB, EMA charts with Daily/Weekly/Monthly tabs
4. **Phase 02 → Train Model**: Show bull/bear/both selector + backtest mode
5. **Phase 03 → Performance**: Show returns chart, Sharpe, MDD
6. **Phase 04 → Observability**: Show drift detection, API latency
7. Open http://localhost:8000/dashboard/ops.html → Show Chat Analytics (recent queries, intent distribution)

---

## Key Talking Points

- **Dual model:** Bull model optimises growth; Bear model protects capital during corrections
- **No Claude API at chat time:** Intent classification is pure cosine similarity on OHLCV data — deterministic, fast, no API cost
- **No database:** Everything is CSV files in `rita_output/` — simple, inspectable, no infrastructure
- **Action space alignment:** Chat `get_strategy_recommendation` uses the same HOLD/HALF/FULL space as the RL model
- **Constraint-driven:** Sharpe > 1.0 AND Max drawdown < 10% — not just return maximisation
- **FnO scope:** BANKNIFTY only in Portfolio Manager — never mixed into the NIFTY RL pipeline

---

## Current Numbers (as of Mar 2026)

- Data: NIFTY 23,114.50 (20-Mar-2026), BNKN 53,427.05
- Backtest Sharpe: 0.826 (8-feature model, Apr 2025 – Feb 2026)
- *9-feature retrain pending — expected to improve Sharpe*
