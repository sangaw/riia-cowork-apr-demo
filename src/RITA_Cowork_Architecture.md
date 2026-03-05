# RITA - Risk Informed Trading Approach
## Cowork + MCP Integration Architecture

---

## Overview

RITA is a systematic trading framework that uses AI-assisted analysis through Claude Desktop + Cowork. The system exposes financial analyst functions via MCP (Model Context Protocol), allowing Claude to orchestrate multi-phase trading decisions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop (Cowork)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Natural Language Interface                          │  │
│  │  "Analyze current market and suggest strategy"       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          MCP Protocol Layer                          │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              RITA MCP Server (Python)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 1: INITIATION                               │    │
│  │  - set_financial_goal(target_return, time_horizon) │    │
│  │  - get_current_portfolio()                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 2: RESEARCH                                 │    │
│  │  - Research Analyst:                               │    │
│  │    • analyze_macro_trends(sectors, regions)        │    │
│  │    • get_supply_demand_analysis(commodity)         │    │
│  │    • project_growth_rates(industry)                │    │
│  │                                                     │    │
│  │  - Sentiment Analyst:                              │    │
│  │    • get_market_sentiment()                        │    │
│  │    • analyze_news_impact(ticker, days=7)           │    │
│  │    • get_investor_positioning()                    │    │
│  │                                                     │    │
│  │  - Technical Analyst:                              │    │
│  │    • get_candlestick_pattern(ticker)               │    │
│  │    • calculate_indicators(ticker, indicators[])    │    │
│  │    • identify_support_resistance(ticker)           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 3: DESIGN (Strategy)                        │    │
│  │  - recommend_strategy(market_conditions)           │    │
│  │  - design_portfolio_allocation(strategy, risk)     │    │
│  │  - identify_opportunities(strategy_type)           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 4: EVALUATION (Scenarios)                   │    │
│  │  - run_scenario_analysis(portfolio, scenarios)     │    │
│  │  - calculate_max_drawdown(portfolio)               │    │
│  │  - suggest_hedging_strategy(risk_level)            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 5: EXECUTION                                │    │
│  │  - execute_trade(ticker, quantity, type)           │    │
│  │  - balance_derivatives(portfolio, hedge_ratio)     │    │
│  │  - get_execution_report()                          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 6: FEEDBACK (Closed Loop)                   │    │
│  │  - log_trade_outcome(trade_id, pnl)                │    │
│  │  - analyze_strategy_performance(period)            │    │
│  │  - update_learning_model(feedback_data)            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                 │
│  - Market Data APIs (Alpha Vantage, Yahoo Finance)          │
│  - News APIs (NewsAPI, Bloomberg)                           │
│  - Historical Trade Database (SQLite/PostgreSQL)            │
│  - Portfolio State (JSON/Database)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## MCP Server Implementation Structure

### Directory Layout

```
rita-mcp-server/
├── pyproject.toml              # Python package config
├── README.md                   # Setup instructions
├── src/
│   └── rita/
│       ├── __init__.py
│       ├── server.py           # Main MCP server entry point
│       ├── analysts/
│       │   ├── __init__.py
│       │   ├── research_analyst.py
│       │   ├── sentiment_analyst.py
│       │   ├── technical_analyst.py
│       │   ├── strategy_analyst.py
│       │   ├── scenario_analyst.py
│       │   ├── execution_analyst.py
│       │   └── outcome_analyst.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── market_data.py
│       │   ├── news_data.py
│       │   └── portfolio_db.py
│       └── utils/
│           ├── __init__.py
│           └── indicators.py
└── claude_desktop_config/
    └── claude_desktop_config.json
```

---

## Phase-by-Phase MCP Function Mapping

### Phase 1: INITIATION

**Purpose:** Define clear, quantitative financial goals

**MCP Functions:**
```python
@mcp.tool()
async def set_financial_goal(
    target_return_pct: float,
    time_horizon_days: int,
    risk_tolerance: str  # "conservative" | "moderate" | "aggressive"
) -> dict:
    """
    Set the financial goal for the trading period.
    
    Returns:
        {
            "goal_id": "uuid",
            "target_return": 15.0,
            "time_horizon": 365,
            "required_monthly_return": 1.17,
            "risk_profile": "moderate"
        }
    """

@mcp.tool()
async def get_benchmark_historical_view(
    benchmark: str = "NIFTY50",
    years: int = 5
) -> dict:
    """
    Get historical performance view of benchmark index.
    
    Returns:
        {
            "benchmark": "NIFTY50",
            "cagr_5y": 13.3,
            "max_drawdown": -18.5,
            "volatility": 15.2
        }
    """
```

---

### Phase 2: RESEARCH

#### 2A. Research Analyst - Macro Trends

**Purpose:** Analyze macro-economic trends, goods flow, demand/supply

**MCP Functions:**
```python
@mcp.tool()
async def analyze_macro_trends(
    sectors: list[str],
    regions: list[str] = ["India", "Global"]
) -> dict:
    """
    Analyze macro-economic trends across sectors and regions.
    
    Returns:
        {
            "sectors": {
                "Technology": {
                    "growth_outlook": "positive",
                    "gdp_contribution": 7.5,
                    "key_drivers": ["AI adoption", "Digital transformation"]
                }
            },
            "regional_outlook": {...}
        }
    """

@mcp.tool()
async def get_supply_demand_analysis(
    commodity_or_sector: str
) -> dict:
    """
    Analyze supply-demand dynamics for commodities or sectors.
    """
```

#### 2B. Sentiment Analyst

**Purpose:** Assess market conditions, news, investor sentiment

**MCP Functions:**
```python
@mcp.tool()
async def get_market_sentiment() -> dict:
    """
    Get current market sentiment indicators.
    
    Returns:
        {
            "overall_sentiment": "bullish",  # "bullish" | "bearish" | "neutral"
            "fear_greed_index": 65,
            "put_call_ratio": 0.85,
            "vix_level": 15.3
        }
    """

@mcp.tool()
async def analyze_news_impact(
    ticker: str,
    days_lookback: int = 7
) -> dict:
    """
    Analyze recent news sentiment impact on a specific ticker.
    """

@mcp.tool()
async def get_investor_positioning() -> dict:
    """
    Get institutional investor positioning data (FII/DII flows).
    """
```

#### 2C. Technical Analyst

**Purpose:** Technical indicators, chart patterns, momentum

**MCP Functions:**
```python
@mcp.tool()
async def get_candlestick_pattern(
    ticker: str,
    timeframe: str = "1D"
) -> dict:
    """
    Identify current candlestick pattern.
    
    Returns:
        {
            "ticker": "RELIANCE",
            "pattern": "bullish_engulfing",
            "reliability": "high",
            "signal": "buy",
            "confirmation_needed": false
        }
    """

@mcp.tool()
async def calculate_indicators(
    ticker: str,
    indicators: list[str]  # ["RSI", "MACD", "Stochastic", "Bollinger"]
) -> dict:
    """
    Calculate multiple technical indicators.
    
    Returns:
        {
            "RSI": {"value": 65, "signal": "neutral"},
            "MACD": {"value": 12.5, "signal": "buy", "histogram": "positive"},
            "Stochastic": {"k": 75, "d": 72, "signal": "overbought"}
        }
    """

@mcp.tool()
async def get_market_trend_analysis(
    ticker: str = "NIFTY50"
) -> dict:
    """
    Classify market trend (Uptrend/Downtrend/Sideways).
    
    Returns:
        {
            "trend": "uptrend",
            "strength": 7.5,  # 0-10 scale
            "duration_days": 45
        }
    """
```

---

### Phase 3: DESIGN (Strategy)

**Purpose:** Decide strategy approach based on research

**MCP Functions:**
```python
@mcp.tool()
async def recommend_strategy(
    market_conditions: dict,  # From research phase
    risk_tolerance: str
) -> dict:
    """
    Recommend investment strategy based on market analysis.
    
    Returns:
        {
            "recommended_strategy": "momentum_investing",
            "reasoning": "Strong uptrend with positive sentiment...",
            "alternative_strategies": ["value_investing"],
            "time_horizon": "medium_term",  # 6-12 months
            "confidence": 0.85
        }
    """

@mcp.tool()
async def design_portfolio_allocation(
    strategy: str,
    total_capital: float,
    risk_level: str
) -> dict:
    """
    Design portfolio allocation based on strategy.
    
    Returns:
        {
            "equity_allocation": 0.70,
            "derivatives_allocation": 0.15,
            "cash_allocation": 0.15,
            "sectors": {
                "Technology": 0.25,
                "Finance": 0.20,
                ...
            }
        }
    """
```

---

### Phase 4: EVALUATION (Scenarios)

**Purpose:** Stress-test portfolio through various scenarios

**MCP Functions:**
```python
@mcp.tool()
async def run_scenario_analysis(
    portfolio: dict,
    scenarios: list[str]  # ["market_crash", "recession", "bull_run"]
) -> dict:
    """
    Run portfolio through multiple scenarios.
    
    Returns:
        {
            "scenarios": {
                "market_crash_20pct": {
                    "portfolio_impact": -15.5,
                    "max_drawdown": -18.2,
                    "recovery_days": 90
                },
                "bull_run_30pct": {
                    "portfolio_impact": 25.3,
                    ...
                }
            }
        }
    """

@mcp.tool()
async def calculate_max_drawdown(
    portfolio: dict
) -> dict:
    """
    Calculate expected maximum drawdown (MDD).
    """

@mcp.tool()
async def suggest_hedging_strategy(
    portfolio: dict,
    max_acceptable_loss_pct: float
) -> dict:
    """
    Suggest derivatives hedging strategy if MDD exceeds threshold.
    
    Returns:
        {
            "hedge_required": true,
            "strategy": "protective_put",
            "instruments": [
                {
                    "type": "put_option",
                    "strike": 18000,
                    "expiry": "2024-03-28",
                    "quantity": 50
                }
            ],
            "hedge_cost": 25000,
            "protection_level": 0.95
        }
    """
```

---

### Phase 5: EXECUTION

**Purpose:** Execute trades with proper risk management

**MCP Functions:**
```python
@mcp.tool()
async def execute_trade(
    ticker: str,
    quantity: int,
    trade_type: str,  # "buy" | "sell" | "buy_option" | "sell_option"
    order_type: str = "market",  # "market" | "limit"
    limit_price: float = None
) -> dict:
    """
    Execute a trade (simulation or real via broker API).
    
    Returns:
        {
            "trade_id": "uuid",
            "status": "executed",
            "fill_price": 2450.50,
            "timestamp": "2024-01-15T10:30:00Z",
            "brokerage": 25.50
        }
    """

@mcp.tool()
async def balance_derivatives(
    portfolio: dict,
    hedge_ratio: float
) -> dict:
    """
    Balance portfolio with derivatives for risk management.
    """

@mcp.tool()
async def get_execution_report() -> dict:
    """
    Get summary of all executed trades in current session.
    """
```

---

### Phase 6: FEEDBACK (Closed Loop)

**Purpose:** Learn from outcomes and improve strategy

**MCP Functions:**
```python
@mcp.tool()
async def log_trade_outcome(
    trade_id: str,
    exit_price: float,
    exit_date: str
) -> dict:
    """
    Log the outcome of a completed trade.
    
    Returns:
        {
            "trade_id": "uuid",
            "entry_price": 2450.50,
            "exit_price": 2680.30,
            "pnl": 11490.0,
            "pnl_pct": 9.38,
            "holding_days": 45,
            "strategy_used": "momentum_investing"
        }
    """

@mcp.tool()
async def analyze_strategy_performance(
    period_days: int = 90,
    strategy: str = None  # None = all strategies
) -> dict:
    """
    Analyze performance of strategies over a period.
    
    Returns:
        {
            "total_trades": 25,
            "win_rate": 0.68,
            "avg_profit_pct": 7.2,
            "avg_loss_pct": -3.5,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.45,
            "strategy_breakdown": {...}
        }
    """

@mcp.tool()
async def get_learning_insights() -> dict:
    """
    Get AI-generated insights from historical performance.
    
    Returns:
        {
            "insights": [
                "Momentum strategy performs 15% better in uptrend markets",
                "Technical indicators show higher accuracy on liquid stocks",
                "Hedging reduces returns by 2% but limits max drawdown to 10%"
            ],
            "recommendations": [
                "Increase momentum allocation in current market conditions",
                "Consider reducing derivatives exposure in low volatility"
            ]
        }
    """
```

---

## Cowork Integration Configuration

### Claude Desktop Config

File: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rita": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/rita-mcp-server",
        "run",
        "rita"
      ],
      "env": {
        "MARKET_DATA_API_KEY": "your_api_key",
        "DATABASE_PATH": "/path/to/rita_data.db"
      }
    }
  }
}
```

---

## Sample Cowork Workflows

### Workflow 1: Daily Market Analysis

**User in Cowork:**
> "RITA: Analyze current market conditions and suggest today's strategy"

**Claude orchestrates:**
1. Calls `get_market_sentiment()`
2. Calls `get_market_trend_analysis()`
3. Calls `get_candlestick_pattern("NIFTY50")`
4. Calls `recommend_strategy()` with combined data
5. Presents analysis in natural language

---

### Workflow 2: Portfolio Review & Rebalance

**User in Cowork:**
> "Review my portfolio performance and suggest rebalancing"

**Claude orchestrates:**
1. Calls `analyze_strategy_performance(period_days=30)`
2. Calls `get_current_portfolio()`
3. Calls `run_scenario_analysis()` with current holdings
4. Calls `design_portfolio_allocation()` for rebalancing
5. Generates rebalancing report with specific trades

---

### Workflow 3: Risk Event Response

**User in Cowork:**
> "Market dropped 5% today. Should I hedge my portfolio?"

**Claude orchestrates:**
1. Calls `get_market_sentiment()` - confirms fear levels
2. Calls `calculate_max_drawdown(portfolio)` - assesses risk
3. Calls `suggest_hedging_strategy()` if MDD exceeds threshold
4. Calls `execute_trade()` for protective puts if approved
5. Updates portfolio protection status

---

## Implementation Checklist

### Phase 1: Core MCP Server Setup
- [ ] Set up Python project structure
- [ ] Implement MCP server with basic health check
- [ ] Test connection with Claude Desktop
- [ ] Implement authentication/security

### Phase 2: Analyst Modules (Sequential)
- [ ] Implement Research Analyst functions
- [ ] Implement Sentiment Analyst functions
- [ ] Implement Technical Analyst functions
- [ ] Implement Strategy Analyst functions
- [ ] Implement Scenario Analyst functions
- [ ] Implement Execution Analyst functions
- [ ] Implement Outcome Analyst functions

### Phase 3: Data Integration
- [ ] Connect to market data APIs
- [ ] Set up news API integration
- [ ] Create portfolio database schema
- [ ] Implement trade logging system

### Phase 4: Testing & Refinement
- [ ] Unit tests for each analyst
- [ ] Integration tests for workflows
- [ ] Backtesting with historical data
- [ ] Paper trading validation

### Phase 5: Documentation
- [ ] API documentation for each MCP function
- [ ] Cowork usage guide for end users
- [ ] Strategy playbooks
- [ ] Troubleshooting guide

---

## Key Benefits of This Architecture

1. **Modular Design**: Each analyst is independent, easy to update
2. **Natural Language Interface**: Users describe intent, Claude orchestrates
3. **Closed Loop Learning**: System improves from every trade
4. **Risk Management**: Automated scenario analysis and hedging
5. **Auditability**: Every decision is logged and traceable
6. **Scalability**: Easy to add new analysts or strategies

---

## Next Steps

1. **Start with Phase 1**: Build minimal MCP server with 2-3 functions
2. **Test in Cowork**: Validate the interaction pattern
3. **Iterate**: Add one analyst at a time
4. **Gather Feedback**: Use outcome data to improve
5. **Scale**: Expand to full RITA workflow

---

**Note:** This architecture allows Claude to act as your intelligent trading assistant, orchestrating complex multi-phase analysis through simple natural language requests in Cowork.
