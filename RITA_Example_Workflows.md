# RITA — Demo Conversation Flows

Practical examples using the 6 standalone MCP tools in Claude Desktop.

---

## Workflow 1: Investment Planning from Scratch

Start a new Claude Desktop conversation. RITA tools appear automatically.

**You:**
> "What returns can I expect from Nifty 50 over 1 year?"

**Claude calls:** `get_return_estimates(period="1y")`

Returns 5 historical percentile scenarios (conservative to best-case), win rate, and a suggested realistic target based on actual Nifty 50 history.

---

**You:**
> "How is the market looking right now?"

**Claude calls:** `get_market_sentiment()`

Returns BULLISH / NEUTRAL / BEARISH consolidated from 5 signals: EMA cross, MACD, RSI, Bollinger Bands, ATR volatility. Score ranges −6 (max bearish) to +6 (max bullish).

---

**You:**
> "What allocation should I take in Nifty 50?"

**Claude calls:** `get_strategy_recommendation()`

Returns HOLD / HALF (50%) / FULL (100%) — the same 3-action space as the RL model. Includes rationale, when to upgrade, when to downgrade.

---

## Workflow 2: Portfolio Sizing

**You:**
> "I have 10 lakh INR to invest. Show me how different strategies would have performed."

**Claude calls:** `get_portfolio_scenarios(portfolio_inr=1000000)`

Compares 4 profiles side-by-side in INR terms:
- Conservative (30% allocation)
- Moderate (60%)
- Aggressive (100%)
- RITA model (RL-driven dynamic allocation)

Shows final portfolio value, return %, Sharpe ratio, max drawdown for each.

---

**You:**
> "What if the market moves 20% up or down from here?"

**Claude calls:** `get_stress_scenarios(portfolio_inr=1000000)`

Stress-tests all 4 profiles plus RITA→HOLD at ±10%, ±20%, flat. Shows ₹ P&L, drawdown breach flag, and RITA's MDD protection mechanism in action.

---

## Workflow 3: Performance Review

**You:**
> "How did RITA perform in the last backtest? What are realistic expectations going forward?"

**Claude calls:** `get_performance_feedback()`

Returns the full backtest outcome: return %, CAGR, Sharpe ratio, max drawdown, trade count, time spent at each allocation (HOLD/HALF/FULL), constraint check (Sharpe > 1.0, MDD < 10%), and 1y/3y forward return expectations with caveats.

---

## Workflow 4: Full Pipeline via MCP

For a complete investment analysis session, Claude can run the full 8-step pipeline:

```
step1_set_financial_goal   → set target return, horizon, risk tolerance
step2_analyze_market       → compute all indicators, classify trend
step3_design_strategy      → select allocation approach
step4_train_model          → train bull / bear / both DDQN models
step5_set_simulation_period→ set backtest window
step6_run_backtest         → run regime-aware backtest
step7_get_results          → generate charts, check constraints
step8_update_goal          → compare actual vs target, revise goal
```

**You:**
> "Run a complete RITA analysis. My goal is 15% return over the next year with moderate risk. Use the existing trained model."

Claude will call all 8 steps in sequence, using `force_retrain=False` to reuse the saved model.

---

## HTML Dashboard Chat

The Market Analysis page in the HTML dashboard also has a built-in RITA chat (no Claude API — pure intent classification):

**Suggested question chips available:**
- "What's the current market trend?"
- "Is now a good time to invest in Nifty?"
- "What are the risk levels right now?"
- "Show me 1-year return estimates"
- "What allocation does the model recommend?"
- "How has RITA performed recently?"
- "What happens in a 10% market crash?"
- "Compare conservative vs aggressive investing"
- "What's the MACD signal saying?"
- "Show me 3-year return scenarios"

Type any question or click a chip. The classifier routes to the appropriate handler (cosine similarity, 20 intents, threshold 0.42).

---

## Tips

**Be specific about amount:**
> "I have ₹5L with high risk tolerance for 6 months" — gives better portfolio scenarios than "I want to invest"

**Chain follow-ups:**
> "What if the market turns sideways next month?" — Claude adapts using the stress scenario tool

**Ask for explanation:**
> "Why HALF allocation and not FULL?" — Claude uses the strategy tool rationale to explain

---

*RITA is a research tool. Not financial advice. Past performance does not guarantee future results.*
