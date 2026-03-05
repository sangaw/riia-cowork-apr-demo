# RITA Cowork Example Workflows

This document shows practical examples of how to use RITA through Claude Desktop's Cowork interface.

---

## Workflow 1: Morning Market Analysis

**Your prompt in Cowork:**
```
RITA: Good morning. Please provide today's market analysis and trading plan.
```

**What Claude does behind the scenes:**
1. Calls `get_benchmark_historical_view()` - Context on long-term trends
2. Calls `get_market_sentiment()` - Current fear/greed levels
3. Calls `get_market_trend_analysis("NIFTY50")` - Is market in uptrend/downtrend?
4. Calls `get_candlestick_pattern("NIFTY50")` - Today's pattern
5. Synthesizes all data into natural language summary

**Expected output:**
```
Good morning! Here's your RITA market analysis for today:

MARKET OVERVIEW
The NIFTY50 is currently at 22,450, continuing its uptrend that started 45 days ago 
with strength of 7.5/10. The market has gained 8.3% over this period.

SENTIMENT
Market sentiment is bullish with a fear/greed index of 65. FII flows show net buying 
of ₹2,500 crores yesterday, and the put-call ratio of 0.85 suggests moderate optimism.

TECHNICAL SIGNALS
Today's candlestick shows a bullish engulfing pattern with high reliability. Key 
technical indicators:
- RSI: 65 (neutral zone)
- MACD: Positive crossover (buy signal)
- Support: 22,000 | Resistance: 23,000

RECOMMENDATION
The technical setup favors continued upside momentum. Consider momentum-based entries 
with stop-loss at 22,000. Watch for potential profit-booking near 23,000 resistance.
```

---

## Workflow 2: Strategy Selection

**Your prompt:**
```
RITA: I have ₹10 lakhs to invest with moderate risk tolerance. What strategy do you 
recommend for the next 6 months?
```

**Claude orchestrates:**
1. Sets financial goal: `set_financial_goal(15.0, 180, "moderate")`
2. Analyzes current conditions with research tools
3. Calls `recommend_strategy(market_conditions, "moderate")`
4. Calls `design_portfolio_allocation("momentum_investing", 1000000, "moderate")`

**Expected output:**
```
Based on current market conditions, here's your customized trading plan:

RECOMMENDED STRATEGY: Momentum Investing
Confidence: 85%

RATIONALE
The market is in a strong uptrend with positive sentiment and bullish technical 
indicators. Momentum strategies historically outperform by 15% in such conditions.

PORTFOLIO ALLOCATION (₹10,00,000)
✓ Equity: ₹7,00,000 (70%)
  - Technology: ₹2,50,000 (25%)
  - Finance: ₹2,00,000 (20%)
  - Energy: ₹1,50,000 (15%)
  - Healthcare: ₹1,00,000 (10%)

✓ Derivatives: ₹1,50,000 (15%)
  - Index options for hedging

✓ Cash: ₹1,50,000 (15%)
  - Reserved for opportunities

EXPECTED OUTCOMES
Target return: 15% over 6 months
Monthly requirement: 2.3%
Max acceptable drawdown: 12%

Would you like me to identify specific stocks in each sector?
```

---

## Workflow 3: Risk Assessment & Hedging

**Your prompt:**
```
RITA: My portfolio is down 8% this week. Should I be worried? What should I do?
```

**Claude orchestrates:**
1. Calls `get_market_sentiment()` - Check if it's general selloff
2. Calls `run_scenario_analysis(your_portfolio, ["market_crash_20pct", "recession"])`
3. Calls `calculate_max_drawdown(your_portfolio)`
4. Calls `suggest_hedging_strategy(your_portfolio, 10.0)`

**Expected output:**
```
CURRENT SITUATION ANALYSIS

Your 8% drawdown is concerning but not critical yet. The broader market is down 6%, 
so you're slightly underperforming.

SCENARIO ANALYSIS
I ran your portfolio through stress tests:

📉 Market Crash (20% drop)
   Your portfolio impact: -15.5%
   Recovery time: ~90 days

📉 Recession Scenario
   Your portfolio impact: -12.8%
   Recovery time: ~120 days

RISK ASSESSMENT
Your maximum drawdown potential: 18.2%
Your risk tolerance: 12% (moderate)
⚠️ You're exposed beyond your comfort level

RECOMMENDED ACTION: Implement Hedging

I suggest a protective put strategy:
- Buy NIFTY 22000 Put (expiry: Mar 28)
- Quantity: 50 contracts
- Cost: ₹7,500
- Protection: Limits downside to 5% more

This reduces potential loss from 18% to 10% while costing only 0.75% of portfolio.

Would you like me to execute this hedge?
```

---

## Workflow 4: Post-Trade Analysis

**Your prompt:**
```
RITA: I just closed my TCS position. I bought at ₹3,500 and sold at ₹3,850. 
Record this trade and analyze my overall performance.
```

**Claude orchestrates:**
1. Calls `log_trade_outcome(trade_id, 3850, "2024-02-08")`
2. Calls `analyze_strategy_performance(90)`
3. Calls `get_learning_insights()`

**Expected output:**
```
TRADE RECORDED ✓

Trade ID: TRD_20240208_103045
Ticker: TCS
Entry: ₹3,500 | Exit: ₹3,850
P&L: ₹35,000 (10% gain)
Holding period: 45 days
Strategy: Momentum investing

PORTFOLIO PERFORMANCE (Last 90 Days)

Total trades: 26
Win rate: 68%
Average profit: 7.2%
Average loss: -3.5%
Profit factor: 2.1
Sharpe ratio: 1.45

🎯 You're beating the benchmark by 3.2%!

LEARNING INSIGHTS

Based on your trading history, I've identified these patterns:

1. ✓ Momentum trades in tech sector show 80% win rate
   → Recommendation: Increase tech allocation

2. ✓ Holding period of 30-60 days yields best results
   → Your 45-day hold was optimal

3. ⚠️ You tend to exit winners too early (avg 10% vs potential 15%)
   → Consider trailing stop-loss instead of fixed targets

4. ✓ Your loss cutting at -5% is disciplined
   → Keep this habit!

NEXT STEPS
Would you like me to scan for similar momentum setups in tech stocks?
```

---

## Workflow 5: Continuous Learning Query

**Your prompt:**
```
RITA: What have I learned about my trading over the past 6 months? 
Any patterns I should know about?
```

**Claude orchestrates:**
1. Calls `analyze_strategy_performance(180)`
2. Calls `get_learning_insights()`

**Expected output:**
```
6-MONTH TRADING RETROSPECTIVE

PERFORMANCE SUMMARY
Total trades: 89
Overall return: 18.3% (vs 13.3% benchmark)
Win rate: 64%
Best month: December (+8.2%)
Worst month: October (-3.1%)

KEY LEARNINGS

✅ WHAT'S WORKING

1. Momentum strategy in uptrends
   Win rate: 72% | Avg gain: 9.5%
   → You've mastered this. Stick with it in bullish markets.

2. Tech sector specialization
   Win rate: 78% vs 55% in other sectors
   → Your edge is in technology stocks. Focus here.

3. Trading high-volume stocks
   Win rate: 68% vs 52% on low-volume
   → Liquidity matters for your strategy.

⚠️ AREAS FOR IMPROVEMENT

1. Value investing attempts
   Win rate: 45% | Avg loss: -4.2%
   → Not your strength. Avoid or paper-trade first.

2. News-driven panic exits
   8 trades exited early on news, missed avg 6% upside
   → Trust your stop-loss, ignore noise.

3. Weekend position sizing
   Trades opened on Fridays have 40% win rate
   → Avoid new positions before weekends.

BEHAVIORAL PATTERNS

📊 You trade best when:
- Market is in confirmed uptrend
- Trading tech stocks with >₹500Cr volume
- Entry on technical breakout with RSI 50-70
- Holding for 30-60 days

📊 You struggle when:
- Trying to catch falling knives (value investing)
- Trading in sideways markets
- Influenced by short-term news

ACTIONABLE RECOMMENDATIONS

1. Create a "Tech Momentum Playbook"
   → Document your winning pattern as a checklist

2. Set calendar reminder: "No Friday entries"
   → Prevent weekend-effect losses

3. Use a trading journal for emotions
   → Track why you exit early, identify triggers

4. Paper-trade value strategies first
   → Build competence before committing capital

Your edge is clear: Tech momentum in uptrends. 
Double down on what works, minimize what doesn't.
```

---

## Workflow 6: Complex Multi-Phase Analysis

**Your prompt:**
```
RITA: Complete analysis for a new investment cycle:
1. Set goal of 20% return in 1 year
2. Analyze current market
3. Recommend strategy
4. Show risk scenarios
5. Suggest initial trades
```

**Claude orchestrates (full RITA workflow):**

**Phase 1: Initiation**
1. `set_financial_goal(20.0, 365, "moderate")`
2. `get_benchmark_historical_view()`

**Phase 2: Research**
3. `get_market_sentiment()`
4. `get_market_trend_analysis()`
5. `analyze_macro_trends(["Technology", "Finance", "Energy"])`
6. `calculate_indicators("NIFTY50", ["RSI", "MACD", "Stochastic"])`

**Phase 3: Design**
7. `recommend_strategy(market_conditions, "moderate")`
8. `design_portfolio_allocation("momentum_investing", 1000000, "moderate")`

**Phase 4: Evaluation**
9. `run_scenario_analysis(portfolio, ["market_crash_20pct", "bull_run_30pct"])`
10. `suggest_hedging_strategy(portfolio, 12.0)`

**Expected comprehensive output:**
```
COMPLETE RITA INVESTMENT PLAN

=== PHASE 1: GOAL SETTING ===

Target: 20% annual return
Time horizon: 12 months
Required monthly return: 1.53%
Risk profile: Moderate
Benchmark (NIFTY50 5Y CAGR): 13.3%
→ Your goal is 50% above benchmark. Achievable but requires active management.

=== PHASE 2: MARKET RESEARCH ===

Current Market State: BULLISH UPTREND
- Trend strength: 7.5/10 (45 days)
- Sentiment: Bullish (Fear/Greed: 65)
- Technical: Buy signals on MACD, neutral RSI
- FII flows: Positive (₹2,500Cr net buying)

Macro Outlook:
✓ Technology: Strong growth, AI adoption driving 25% YoY
✓ Finance: Stable, credit growth at 15%
⚠ Energy: Mixed, oil volatility risk

=== PHASE 3: STRATEGY DESIGN ===

RECOMMENDED: Momentum Investing
Confidence: 85%

Why? Strong uptrend + positive sentiment = ideal for momentum

Portfolio Allocation (₹10,00,000):
- Equity: ₹7,00,000 (70%)
  * Tech: ₹2,50,000 (25%)
  * Finance: ₹2,00,000 (20%)
  * Energy: ₹1,50,000 (15%)
  * Healthcare: ₹1,00,000 (10%)
- Derivatives: ₹1,50,000 (15%) - hedging
- Cash: ₹1,50,000 (15%) - opportunities

=== PHASE 4: RISK EVALUATION ===

Scenario Analysis:

📈 Bull Run (+30%)
   Your portfolio: +25.3%
   Result: Exceed goal ✓

📉 Market Crash (-20%)
   Your portfolio: -15.5%
   Max drawdown: -18.2%
   Recovery: 90 days
   
⚠️ RISK ALERT: Max drawdown exceeds 12% threshold

Recommended Hedge:
- Protective puts on NIFTY
- Cost: ₹7,500 (0.75%)
- Reduces max loss to -10%

=== PHASE 5: EXECUTION PLAN ===

Immediate Actions (Week 1):

1. TECH SECTOR (₹2,50,000)
   - TCS: ₹80,000 (breakout at ₹3,850)
   - Infosys: ₹70,000 (momentum)
   - HCL Tech: ₹60,000
   - Tech Mahindra: ₹40,000

2. FINANCE SECTOR (₹2,00,000)
   - HDFC Bank: ₹80,000
   - ICICI Bank: ₹70,000
   - Axis Bank: ₹50,000

3. HEDGING
   - Buy NIFTY 22000 Put (Mar expiry)
   - Quantity: 50 lots
   - Cost: ₹7,500

Execution timeline:
Week 1: Deploy 50% (₹5,00,000)
Week 2-3: Deploy remaining 20% (₹2,00,000) on dips
Reserve: 15% cash for opportunities

Stop-loss: 5% per position
Target: 15-20% per position
Review: Every 30 days

=== SUCCESS METRICS ===

Track these monthly:
✓ Portfolio return vs NIFTY50
✓ Win rate (target: >65%)
✓ Max drawdown (limit: <12%)
✓ Sharpe ratio (target: >1.2)

=== NEXT STEPS ===

Ready to begin? I can:
1. Execute these trades (simulation mode)
2. Set up weekly review reminders
3. Create a trading journal template
4. Monitor positions and alert on stop-loss triggers

Which would you like to start with?
```

---

## Tips for Effective RITA Conversations

### 1. Be Specific
❌ "What should I do?"
✅ "I have ₹5L to invest with high risk tolerance for 6 months. Suggest strategy."

### 2. Provide Context
Include your:
- Capital amount
- Risk tolerance
- Time horizon
- Current positions (if any)

### 3. Ask Follow-ups
RITA learns from conversation:
```
User: "Why did you recommend momentum over value?"
RITA: [Explains based on current market conditions]
User: "What if market turns sideways?"
RITA: [Adjusts recommendation]
```

### 4. Use Natural Language
You don't need technical terms:
```
"Should I be worried about my losses?"
vs
"Run scenario analysis with 20% downside Monte Carlo simulation"

Both work! RITA translates to appropriate technical analysis.
```

---

## Common Questions

**Q: Can RITA execute real trades?**
A: Currently in simulation mode. Real broker integration coming soon (Zerodha, IB).

**Q: How often should I consult RITA?**
A: Daily for market view, weekly for strategy review, monthly for performance analysis.

**Q: Does RITA replace my financial advisor?**
A: No. RITA is a research tool. Always consult qualified professionals for advice.

**Q: Can I customize RITA's risk parameters?**
A: Yes! Your risk tolerance and preferences are stored in the database.

---

**Happy trading with RITA! 📊📈**
