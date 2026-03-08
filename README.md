# RITA MCP Server
**Risk Informed Trading Approach - Claude Desktop Integration**

An intelligent trading assistant that exposes systematic trading analysis through MCP (Model Context Protocol) to Claude Desktop's Cowork feature.

## Quick Start

### 1. Installation

```bash
# Clone or create the project directory
mkdir rita-mcp-server
cd rita-mcp-server

# Install using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configure Claude Desktop

Edit your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add RITA to your MCP servers:

```json
{
  "mcpServers": {
    "rita": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/rita-mcp-server",
        "run",
        "rita"
      ],
      "env": {
        "MARKET_DATA_API_KEY": "your_api_key_here",
        "DATABASE_PATH": "/path/to/rita_data.db"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

Close and reopen Claude Desktop. You should see RITA tools available in Cowork.

## Usage in Cowork

### Example 1: Daily Market Analysis
```
RITA: What's the current market sentiment and trend?
```

Claude will:
- Call `get_market_sentiment()`
- Call `get_market_trend_analysis()`
- Synthesize a natural language summary

### Example 2: Strategy Recommendation
```
RITA: Analyze the market and suggest a trading strategy for the next month
```

Claude will:
- Gather macro trends
- Check sentiment and technical indicators
- Recommend strategy with reasoning

### Example 3: Portfolio Risk Assessment
```
RITA: I have 70% equity, 15% derivatives, 15% cash. Run scenario analysis.
```

Claude will:
- Run scenario simulations
- Calculate max drawdown
- Suggest hedging if needed

## Project Structure

```
rita-mcp-server/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── src/
│   └── rita/
│       ├── __init__.py
│       ├── server.py       # Main MCP server
│       ├── analysts/       # Analyst modules
│       │   ├── research_analyst.py
│       │   ├── sentiment_analyst.py
│       │   ├── technical_analyst.py
│       │   ├── strategy_analyst.py
│       │   ├── scenario_analyst.py
│       │   ├── execution_analyst.py
│       │   └── outcome_analyst.py
│       ├── data/           # Data layer
│       │   ├── market_data.py
│       │   ├── news_data.py
│       │   └── portfolio_db.py
│       └── utils/
│           └── indicators.py
```

## RITA Workflow Phases

### Phase 1: Initiation
- Set financial goals (target return, time horizon)
- Get benchmark performance view

### Phase 2: Research
- **Research Analyst**: Macro trends, supply/demand
- **Sentiment Analyst**: Market sentiment, news analysis
- **Technical Analyst**: Chart patterns, indicators

### Phase 3: Design
- **Strategy Analyst**: Recommend investment approach
- Design portfolio allocation

### Phase 4: Evaluation
- **Scenario Analyst**: Stress test portfolio
- Suggest hedging strategies

### Phase 5: Execution
- **Execution Analyst**: Execute trades
- Balance derivatives positions

### Phase 6: Feedback
- **Outcome Analyst**: Log trade results
- Generate learning insights
- Close the loop for continuous improvement

## Available MCP Tools

### Initiation
- `set_financial_goal` - Define trading objectives
- `get_benchmark_historical_view` - Historical benchmark data

### Research
- `analyze_macro_trends` - Economic and sector analysis
- `get_market_sentiment` - Fear/greed indicators
- `analyze_news_impact` - News sentiment analysis
- `get_market_trend_analysis` - Trend classification
- `get_candlestick_pattern` - Pattern recognition
- `calculate_indicators` - RSI, MACD, Stochastic, Bollinger

### Design
- `recommend_strategy` - Strategy recommendation
- `design_portfolio_allocation` - Asset allocation

### Evaluation
- `run_scenario_analysis` - Portfolio stress testing
- `suggest_hedging_strategy` - Risk mitigation

### Execution
- `execute_trade` - Trade execution (simulation or real)

### Feedback
- `log_trade_outcome` - Record trade results
- `analyze_strategy_performance` - Performance metrics
- `get_learning_insights` - AI-generated insights

## Configuration

### Environment Variables

- `MARKET_DATA_API_KEY` - API key for market data provider
- `DATABASE_PATH` - Path to SQLite database for trade history
- `NEWS_API_KEY` - (Optional) News API key
- `BROKER_API_KEY` - (Optional) Broker API for live trading

### Data Sources

Currently supports:
- Yahoo Finance (via `yfinance`) - Market data
- Alpha Vantage - Alternative market data
- NewsAPI - News sentiment (requires key)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff check src/
```

### Adding New Analysts

1. Create new analyst module in `src/rita/analysts/`
2. Define MCP tools in `server.py`
3. Implement handler functions
4. Update documentation

## Roadmap

- [ ] Implement all analyst modules with real data
- [ ] Add backtesting framework
- [ ] Integration with broker APIs (Zerodha, Interactive Brokers)
- [ ] Machine learning models for strategy optimization
- [ ] Real-time alerts and notifications
- [ ] Multi-asset class support (crypto, forex, commodities)
- [ ] Portfolio visualization dashboard

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Disclaimer

⚠️ **IMPORTANT**: This is a research and educational tool. 

- Not financial advice
- Use at your own risk
- Past performance does not guarantee future results
- Always do your own research
- Consider consulting a qualified financial advisor

## Support

For issues or questions:
- GitHub Issues: [your-repo-url]
- Email: your.email@example.com

---

**Happy Trading with RITA! 📈**


The memory file at C:\Users\Sandeep\.claude\projects\...\memory\MEMORY.md has the full context for the next session


All 3 phases committed to GitHub: https://github.com/sangaw/riia-cowork-apr-demo.git

  When you come back, start with:
  cd C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo
  .\activate-env.ps1
  python verify.py          # confirms everything works in ~10s

  Context is preserved in:
  - project-report.md — full architecture, API reference, "Continuing the Project" section
  - MEMORY.md — Claude session memory auto-loaded next time