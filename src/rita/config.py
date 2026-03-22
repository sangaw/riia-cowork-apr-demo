"""
RITA — Central Configuration

All runtime settings are read from environment variables.
Set them in activate-env.ps1 (local) or your CI/CD environment.

Environment variables
---------------------
NIFTY_CSV_PATH   Path to the Nifty 50 OHLCV CSV (required for any data work)
OUTPUT_DIR       Where model files, CSVs, plots are written  (default: ./rita_output)
RITA_INPUT_DIR   Where raw NSE-format CSVs are dropped        (default: ./rita_input)
RITA_API_PORT    Port for the FastAPI server                   (default: 8000)
RITA_UI_PORT     Port for the Streamlit UI                     (default: 8501)
RITA_API_URL     Base URL for the REST API                     (default: http://localhost:{RITA_API_PORT})
PORTFOLIO_API_KEY  Optional API key for the portfolio endpoint (default: empty = open)
PYTHON_ENV       Environment name (development / production)   (default: development)
"""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from project root if present (python-dotenv optional)."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv()
    except ImportError:
        # Fall back to a minimal parser so the project works without python-dotenv
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

# ── Data paths ────────────────────────────────────────────────────────────────

NIFTY_CSV_PATH: str = os.getenv("NIFTY_CSV_PATH", "")
"""Absolute path to the base Nifty 50 OHLCV CSV. Must be set via env var."""

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./rita_output")
"""Directory for model weights, output CSVs, and plots."""

INPUT_DIR: str = os.getenv("RITA_INPUT_DIR", "./rita_input")
"""Directory for raw NSE-format OHLCV CSVs to be ingested via Data Prep."""

# ── Server ────────────────────────────────────────────────────────────────────

API_PORT: int = int(os.getenv("RITA_API_PORT", "8000"))
"""Port for the FastAPI REST API + HTML dashboard server."""

UI_PORT: int = int(os.getenv("RITA_UI_PORT", "8501"))
"""Default starting port for the Streamlit UI (auto-increments if busy)."""

API_BASE_URL: str = os.getenv("RITA_API_URL", f"http://localhost:{API_PORT}")
"""Base URL used by the Streamlit DevOps tab to ping the REST API."""

# ── Security ──────────────────────────────────────────────────────────────────

PORTFOLIO_API_KEY: str = os.getenv("PORTFOLIO_API_KEY", "")
"""Optional API key for the /api/v1/portfolio/summary endpoint.
Leave empty (default) for open local-dev access."""

# ── Runtime ───────────────────────────────────────────────────────────────────

PYTHON_ENV: str = os.getenv("PYTHON_ENV", "development")
