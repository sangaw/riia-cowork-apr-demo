# RITA Setup Guide

## Prerequisites

- **Python 3.10+** with the `rita` package installed (editable install via `pip install -e .`)
- **Claude Desktop:** For MCP integration
- **Nifty 50 CSV:** OHLCV data file; path set via `NIFTY_CSV_PATH` env var

## 1. Activate Environment

```powershell
cd C:\path\to\rita-cowork-demo
. .\activate-env.ps1
```

The `activate-env.ps1` script sets `NIFTY_CSV_PATH`, `OUTPUT_DIR`, and `PYTHONPATH`.
Copy `activate-env.ps1.example` to `activate-env.ps1` and fill in your local paths if missing.

## 2. Verify Installation

```powershell
# Quick check — steps 1-3, no training (~10s)
python verify.py
```

Expected: all 3 steps print OK, no errors.

## 3. Run the APIs

```powershell
# FastAPI + HTML dashboard (port 8000)
python run_api.py

# Streamlit data-science UI (port 8501)
python run_ui.py
```

| URL | What you see |
|-----|-------------|
| http://localhost:8000/dashboard/ | Landing page (4 cards) |
| http://localhost:8000/dashboard/rita.html | Main RITA app |
| http://localhost:8000/dashboard/fno.html | FnO Portfolio Manager |
| http://localhost:8000/dashboard/ops.html | Operations Portal |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:8501 | Streamlit data-science UI |

## 4. Run the Full Pipeline (optional)

```powershell
python run_pipeline.py     # reuses saved model — fast (~30s)
```

Or use the Streamlit UI — click **▶ Re-use Model Pipeline**.

## 5. Run Tests

```powershell
pytest tests/                          # all 39 tests (requires NIFTY_CSV_PATH)
pytest tests/ -k "not APIEndpoints"   # 29 unit tests only (no CSV needed, ~5s)
```

## 6. Claude Desktop MCP Setup

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rita-cowork": {
      "command": "C:\\path\\to\\your\\python.exe",
      "args": ["-m", "rita.interfaces.mcp_server"],
      "env": {
        "NIFTY_CSV_PATH": "C:\\path\\to\\rita-cowork-demo\\rita_output\\nifty_merged.csv",
        "OUTPUT_DIR": "C:\\path\\to\\rita-cowork-demo\\rita_output",
        "PYTHONPATH": "C:\\path\\to\\rita-cowork-demo\\src",
        "PYTHON_ENV": "development"
      }
    }
  }
}
```

A template is at `config/claude_desktop_config.json.example`. Restart Claude Desktop after editing.

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `NIFTY_CSV_PATH` | Base OHLCV CSV | `C:\...\rita_output\nifty_merged.csv` |
| `OUTPUT_DIR` | Model + output directory | `C:\...\rita_output` |
| `PYTHONPATH` | src directory for rita package | `C:\...\rita-cowork-demo\src` |
| `RITA_INPUT_DIR` | (optional) Input CSV directory | `C:\...\rita_input` |

## Data

- Drop NSE-format OHLCV CSVs into `rita_input/`
- Use **Data Prep** page in the HTML dashboard or Streamlit to merge them
- **Warning:** Do not run Data Prep while `banknifty_manual.csv` is in `rita_input/` — it will be mixed into the NIFTY series
- NSE export numbers must be clean decimals (no Indian comma formatting like `23,114.50`)

## Troubleshooting

**`rita` package not found:**
```powershell
pip install -e .    # run from project root with your Python env active
```

**Claude Desktop doesn't see RITA tools:**
1. Check absolute paths in `claude_desktop_config.json`
2. Verify the Python binary path you configured actually exists
3. Restart Claude Desktop fully (close and reopen)

**API returns 500 errors on performance/risk pages:**
- Run the pipeline first (Re-use Model Pipeline) to generate output CSVs
- These pages require `backtest_daily.csv`, `performance_summary.csv`, etc. in `rita_output/`

**Model retraining:**
- The trained model (`rita_ddqn_model.zip`) is not in git
- It is regenerated automatically if missing (~6 min for 500k timesteps)
- Copy from a previous run into `rita_output/` to skip retraining
