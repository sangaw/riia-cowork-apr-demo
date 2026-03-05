"""
RITA Core — Data Loader
Loads and preprocesses Nifty 50 OHLCV data from the local merged CSV.

Expected CSV columns: date, open, high, low, close, shares traded, turnover (₹ cr)
Date format: "1999-07-01 00:00:00+05:30" (timezone-aware IST)
"""

import math
import numpy as np
import pandas as pd


# Fixed data split boundaries
TRAIN_START = "2010-01-01"
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2024-12-31"
BACKTEST_START = "2025-01-01"

RISK_FREE_RATE = 0.07  # India 10Y government bond yield


def load_nifty_csv(csv_path: str) -> pd.DataFrame:
    """
    Load Nifty 50 data from the merged CSV file.

    Returns a DataFrame with DatetimeIndex and columns:
    Open, High, Low, Close, Volume
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse and normalize the date column (strip timezone)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("date")
    df.index.name = "Date"

    # Rename to standard OHLCV names
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "shares traded": "Volume",
    }
    df = df.rename(columns=rename_map)

    # Drop unused columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    # Sort and validate
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Close"])

    if len(df) == 0:
        raise ValueError(f"No valid data found in {csv_path}")

    return df


def get_date_slice(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return rows between start_date and end_date inclusive (YYYY-MM-DD strings)."""
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    result = df.loc[mask]
    if len(result) == 0:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    return result


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed training slice: 2010-01-01 to 2022-12-31."""
    return get_date_slice(df, TRAIN_START, TRAIN_END)


def get_validation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed validation slice: 2023-01-01 to 2024-12-31."""
    return get_date_slice(df, VALIDATION_START, VALIDATION_END)


def get_backtest_data(df: pd.DataFrame, start: str = BACKTEST_START, end: str = None) -> pd.DataFrame:
    """Return the backtest slice (default: 2025 onwards)."""
    if end is None:
        end = df.index.max().strftime("%Y-%m-%d")
    return get_date_slice(df, start, end)


def get_historical_stats(df: pd.DataFrame) -> dict:
    """
    Calculate full-history performance statistics from the loaded DataFrame.
    Used in Step 1 to anchor the financial goal to real data.
    """
    daily_returns = df["Close"].pct_change().dropna()
    years = len(df) / 252.0

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = (end_price / start_price) - 1
    cagr = (end_price / start_price) ** (1 / years) - 1

    annual_vol = daily_returns.std() * math.sqrt(252)

    sharpe = (daily_returns.mean() - RISK_FREE_RATE / 252) / daily_returns.std() * math.sqrt(252)

    # Rolling max drawdown
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Best and worst annual returns
    annual_rets = df["Close"].resample("YE").last().pct_change().dropna()

    return {
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "total_days": len(df),
        "years": round(years, 1),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annual_volatility_pct": round(annual_vol * 100, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "best_year_pct": round(float(annual_rets.max() * 100), 2),
        "worst_year_pct": round(float(annual_rets.min() * 100), 2),
        "avg_annual_return_pct": round(float(annual_rets.mean() * 100), 2),
    }
