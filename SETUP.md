# RITA Setup Guide for Windows

## Prerequisites

1. **Python 3.10+** - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   - Verify installation:
     ```powershell
     python --version
     ```

2. **uv Package Manager** (recommended) - Faster than pip
   ```powershell
   pip install uv
   ```
   Or download from [astral.sh/uv](https://astral.sh/uv)

3. **Claude Desktop** - Download from [claude.ai](https://claude.ai)

## Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo
```

### Step 2: Install Dependencies

Using `uv` (recommended):
```powershell
uv pip install -e .
```

Or using `pip`:
```powershell
pip install -e .
```

This installs the RITA MCP server and all dependencies in development mode.

### Step 3: Verify Installation

```powershell
# Check that the rita command is available
rita --help
```

If this works, you're ready to configure Claude Desktop.

## Configuration for Claude Desktop

### Step 1: Find Claude Config File

The config file location is:
```
%APPDATA%\Claude\claude_desktop_config.json
```

Or manually navigate to:
- Press `Win + R`
- Type: `%APPDATA%\Claude\`
- Open `claude_desktop_config.json` with a text editor

### Step 2: Update Config File

The project's config file is already set up at:
```
config\claude_desktop_config.json
```

**Option A: Copy the entire config**
```json
{
  "mcpServers": {
    "rita": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\Sandeep\\Documents\\Work\\code\\poc\\rita-cowork-demo",
        "run",
        "rita"
      ],
      "env": {
        "MARKET_DATA_API_KEY": "demo_mode",
        "DATABASE_PATH": "C:\\Users\\Sandeep\\.rita\\trades.db",
        "NEWS_API_KEY": "optional_newsapi_key",
        "PYTHON_ENV": "development"
      }
    }
  }
}
```

**Option B: Update path in your existing config**
If you already have other MCP servers configured, just add the `rita` entry under `mcpServers`.

### Step 3: Restart Claude Desktop

1. Close Claude Desktop completely
2. Reopen Claude Desktop
3. You should see RITA tools available in Cowork

## Using RITA in Claude Desktop

Once configured, you can use RITA in Claude Desktop's Cowork feature:

### Example 1: Daily Market Analysis
```
RITA: What's the current market sentiment and trend for NIFTY50?
```

### Example 2: Portfolio Analysis
```
RITA: I have ₹10 lakhs to invest with moderate risk tolerance. What strategy do you recommend?
```

### Example 3: Risk Assessment
```
RITA: Run scenario analysis on my portfolio across market crash and bull run scenarios.
```

Claude will automatically call RITA's analysis tools and synthesize the results into actionable insights.

## Demo Mode vs Real Data

### Current Setup (Demo Mode)
- **MARKET_DATA_API_KEY**: `demo_mode`
- Returns realistic, randomized mock data
- Perfect for demo and testing
- No external API calls

### To Enable Real Data (Future)
1. Get API keys from:
   - [Alpha Vantage](https://www.alphavantage.co/) - Stock market data
   - [NewsAPI](https://newsapi.org/) - News sentiment
   - Your broker's API (Zerodha, ICICI Direct, etc.)

2. Update `.env` file:
   ```
   MARKET_DATA_API_KEY=your_alpha_vantage_key
   NEWS_API_KEY=your_newsapi_key
   ```

3. Restart Claude Desktop

## Troubleshooting

### "rita command not found"
- Run: `uv pip install -e .` again
- Make sure you're in the project directory

### "Claude Desktop doesn't see RITA"
1. Check the path in `claude_desktop_config.json` - use absolute paths
2. Restart Claude Desktop fully (close and reopen)
3. Check Claude Desktop logs for errors

### Port/Connection Issues
- Make sure no other MCP servers are using the same stdio connection
- Try restarting Claude Desktop

### Mock Data Not Generating
- All handlers are designed to work in demo mode
- Check console output for any error messages
- Verify `MARKET_DATA_API_KEY=demo_mode` in your config

## Project Structure

```
rita-cowork-demo/
├── config/
│   └── claude_desktop_config.json    # MCP server config for Claude
├── src/
│   └── rita/
│       ├── __init__.py              # Package initialization
│       └── server.py                # Main MCP server with all tools
├── pyproject.toml                    # Python project configuration
├── README.md                         # Project overview
├── RITA_Example_Workflows.md         # Use case examples
├── .env.example                      # Environment variables template
└── SETUP.md                          # This file

```

## Next Steps

1. ✅ Start Claude Desktop
2. ✅ Try example prompts from RITA_Example_Workflows.md
3. ✅ Test different trading scenarios
4. ⏭️ Integrate real market data APIs when ready

## Need Help?

- Check `RITA_Example_Workflows.md` for detailed use case examples
- Review `RITA_Cowork_Architecture.md` for system design
- Check the handlers in `src/rita/server.py` for implementation details

Happy trading analysis! 📈
