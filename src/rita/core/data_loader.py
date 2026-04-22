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

    # Parse and normalize the date column (strip timezone if present)
    sample = str(df["date"].iloc[0])
    if "+" in sample or len(sample) > 12:
        # Timezone-aware format (e.g. Nifty: "1999-07-01 00:00:00+05:30")
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        # Plain ISO date (e.g. ASML: "2001-03-09")
        df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = "Date"

    # Rename to standard OHLCV names
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "shares traded": "Volume",  # Nifty NSE column name
        "volume": "Volume",         # Generic / international column name
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


def load_prepared_csv(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a prepared instrument CSV that may already contain pre-computed indicator columns.

    Detection: if 'rsi_14' is present the CSV is fully-featured; the indicator
    DataFrame is returned as-is.  Otherwise falls back to load_nifty_csv() and
    the caller is responsible for running calculate_indicators().

    Returns:
        (raw_df, feat_df)
        - raw_df  : OHLCV-only DataFrame (DatetimeIndex)
        - feat_df : DataFrame with all indicator columns (DatetimeIndex).
                    Equals raw_df when the CSV is OHLCV-only (no indicators pre-computed).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect date column (case-insensitive)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"No 'date' column found in {csv_path}")

    # Parse dates
    sample = str(df[date_col].iloc[0])
    if "+" in sample or len(sample) > 12:
        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Normalise OHLCV column names
    rename_map = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "shares traded": "Volume", "volume": "Volume",
    }
    df = df.rename(columns={c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map})

    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    raw_df = df[ohlcv_cols].dropna(subset=["Close"])

    if len(raw_df) == 0:
        raise ValueError(f"No valid data found in {csv_path}")

    if "rsi_14" in df.columns:
        # Fully-featured CSV — use pre-computed indicators, drop rows missing Close
        feat_df = df.dropna(subset=["Close"])
        return raw_df, feat_df

    # OHLCV-only — caller must call calculate_indicators()
    return raw_df, raw_df


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


def parse_period_to_days(period) -> int:
    """
    Convert a period string or integer to calendar days.

    Accepted formats: int, or strings like "1d", "2w", "6m", "1y", "3y".
    Also accepts plain digit strings like "365".
    Range: 1 day to 1095 days (3 years).
    """
    if isinstance(period, int):
        days = period
    elif isinstance(period, str):
        period = period.strip().lower()
        if period.isdigit():
            days = int(period)
        elif period.endswith("d"):
            days = int(period[:-1])
        elif period.endswith("w"):
            days = int(period[:-1]) * 7
        elif period.endswith("m"):
            days = int(period[:-1]) * 30
        elif period.endswith("y"):
            days = int(period[:-1]) * 365
        else:
            raise ValueError(
                f"Cannot parse period '{period}'. "
                "Use formats like '1d', '2w', '6m', '1y', '3y', or a plain number of days."
            )
    else:
        raise ValueError(f"period must be a string or int, got {type(period)}")

    if days < 1:
        raise ValueError("period must be at least 1 day.")
    if days > 1095:
        raise ValueError("period cannot exceed 3 years (1095 days).")
    return days


def get_period_return_estimates(df: pd.DataFrame, period_days: int) -> dict:
    """
    Compute rolling-window return distribution for a given investment horizon.

    Slides a window of `period_days` calendar days across the full history and
    collects the total return for each window.  Returns percentile-based
    scenarios so callers can anchor expectations to real Nifty 50 data.
    """
    close = df["Close"]

    # Convert calendar days → approximate trading days (252 / 365)
    trading_days = max(1, round(period_days * 252 / 365))

    # All rolling total returns (in %)
    rolling_returns = close.pct_change(trading_days).dropna() * 100

    if len(rolling_returns) < 10:
        raise ValueError(
            f"Not enough historical data to estimate {period_days}-day returns "
            f"(only {len(rolling_returns)} windows available)."
        )

    p10 = float(np.percentile(rolling_returns, 10))
    p25 = float(np.percentile(rolling_returns, 25))
    p50 = float(np.percentile(rolling_returns, 50))
    p75 = float(np.percentile(rolling_returns, 75))
    p90 = float(np.percentile(rolling_returns, 90))

    win_rate = float((rolling_returns > 0).sum() / len(rolling_returns) * 100)

    years = period_days / 365.0

    def _annualize(total_pct: float) -> float:
        """Express a total return as annualised CAGR when horizon >= 1 year."""
        total = total_pct / 100.0
        if years >= 1.0 and total > -1.0:
            return round(((1.0 + total) ** (1.0 / years) - 1.0) * 100.0, 2)
        return round(total_pct, 2)

    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date   = df.index.max().strftime("%Y-%m-%d")

    return {
        "period_days": period_days,
        "trading_days_used": trading_days,
        "sample_windows": len(rolling_returns),
        "data_range": f"{start_date} to {end_date}",
        "win_rate_pct": round(win_rate, 1),
        "scenarios": {
            "conservative": {"total_return_pct": round(p10, 2), "annualized_pct": _annualize(p10)},
            "cautious":     {"total_return_pct": round(p25, 2), "annualized_pct": _annualize(p25)},
            "median":       {"total_return_pct": round(p50, 2), "annualized_pct": _annualize(p50)},
            "optimistic":   {"total_return_pct": round(p75, 2), "annualized_pct": _annualize(p75)},
            "best_case":    {"total_return_pct": round(p90, 2), "annualized_pct": _annualize(p90)},
        },
        "suggested_target_pct": round(p10, 2),
        "note": (
            f"Conservative estimate based on 10th percentile of {len(rolling_returns)} "
            f"rolling {period_days}-day windows ({start_date} to {end_date})."
        ),
    }


def prepare_data(input_dir: str, base_csv: str, output_csv: str) -> dict:
    """
    Merge new NSE-format CSV files from input_dir with the existing base merged CSV.

    Input files use the standard NSE export format:
        columns  : Date, Open, High, Low, Close, Shares Traded, Turnover (₹ Cr)
        date fmt : DD-MMM-YYYY  (e.g. 13-MAR-2026)

    The result is saved to output_csv in the same format as the base merged CSV
    (lowercase columns, IST-aware dates) so load_nifty_csv() can read it unchanged.

    Priority for existing data:
        1. output_csv  — if it already exists (previous prepare run)
        2. base_csv    — original merged.csv from external project
        3. input files only — if neither exists

    Returns a summary dict with: status, files_processed, rows_added,
    total_rows, date_from, date_to, output_csv.
    """
    import glob as _glob
    import os as _os

    # ── Collect new input files ──────────────────────────────
    input_files = sorted(_glob.glob(_os.path.join(input_dir, "*.csv")))
    if not input_files:
        return {
            "status": "no_input_files",
            "files_found": 0,
            "rows_added": 0,
            "message": f"No CSV files found in {input_dir}",
        }

    new_dfs = []
    skipped = []
    for f in input_files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip().str.lower()
            if "date" not in df.columns:
                skipped.append(_os.path.basename(f))
                continue
            new_dfs.append(df)
        except Exception as exc:
            skipped.append(f"{_os.path.basename(f)} ({exc})")

    if not new_dfs:
        return {"status": "error", "message": "Could not read any input files", "skipped": skipped}

    new_data = pd.concat(new_dfs, ignore_index=True)

    # Parse dates from DD-MMM-YYYY (NSE export) → timezone-aware IST
    new_data["date"] = pd.to_datetime(new_data["date"].astype(str), errors="coerce")
    new_data = new_data.dropna(subset=["date"])
    if new_data["date"].dt.tz is None:
        new_data["date"] = new_data["date"].dt.tz_localize("Asia/Kolkata")
    else:
        new_data["date"] = new_data["date"].dt.tz_convert("Asia/Kolkata")

    # ── Load existing base data ──────────────────────────────
    # Prefer output_csv (already extended) over base_csv (original)
    existing_source = output_csv if _os.path.exists(output_csv) else base_csv
    rows_before = 0

    if _os.path.exists(existing_source):
        existing = pd.read_csv(existing_source)
        existing.columns = existing.columns.str.strip().str.lower()
        existing["date"] = pd.to_datetime(existing["date"].astype(str), errors="coerce")
        existing = existing.dropna(subset=["date"])
        if existing["date"].dt.tz is None:
            existing["date"] = existing["date"].dt.tz_localize("Asia/Kolkata")
        else:
            existing["date"] = existing["date"].dt.tz_convert("Asia/Kolkata")
        rows_before = len(existing)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # ── Dedup, sort, save ────────────────────────────────────
    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )

    output_dir = _os.path.dirname(output_csv)
    if output_dir:
        _os.makedirs(output_dir, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    rows_added = len(combined) - rows_before
    return {
        "status": "ok",
        "files_processed": len(new_dfs),
        "skipped": skipped,
        "rows_added": rows_added,
        "total_rows": len(combined),
        "date_from": str(combined["date"].iloc[0])[:10],
        "date_to":   str(combined["date"].iloc[-1])[:10],
        "output_csv": output_csv,
    }


def get_bear_episodes(df: pd.DataFrame, min_duration_days: int = 20, buffer_days: int = 10) -> pd.DataFrame:
    """
    Extract bear-market episodes from the training slice (2010-2022) for the bear model.

    A bear episode is defined as a contiguous period where ema_26 / ema_50 < 0.99.
    Episodes shorter than `min_duration_days` trading days are discarded (noise).
    `buffer_days` trading days are prepended to each episode so the model sees the
    run-up before the drawdown begins.

    Args:
        df: Full DataFrame already processed by calculate_indicators()
        min_duration_days: Minimum trading days for an episode to be included (default 20)
        buffer_days: Trading days of context prepended before bear start (default 10)

    Returns:
        DataFrame containing only bear-episode rows, sorted by date.
        Rows retain their original index so they can be used directly by NiftyTradingEnv.
    """
    # Work within training slice only
    train = df[(df.index >= pd.Timestamp(TRAIN_START)) & (df.index <= pd.Timestamp(TRAIN_END))].copy()
    train = train.dropna(subset=["ema_26", "ema_50"])

    if train.empty:
        raise ValueError("No training data available after dropna on ema columns")

    bear_mask = (train["ema_26"] / train["ema_50"]) < 0.99
    indices = train.index.tolist()

    # Find contiguous bear runs
    episodes = []
    in_bear = False
    start_idx = None
    for i, idx in enumerate(indices):
        if bear_mask[idx] and not in_bear:
            in_bear = True
            start_idx = i
        elif not bear_mask[idx] and in_bear:
            end_idx = i - 1
            duration = end_idx - start_idx + 1
            if duration >= min_duration_days:
                buf_start = max(0, start_idx - buffer_days)
                episodes.append(indices[buf_start: end_idx + 1])
            in_bear = False
    # Handle case where bear run extends to end
    if in_bear:
        end_idx = len(indices) - 1
        duration = end_idx - start_idx + 1
        if duration >= min_duration_days:
            buf_start = max(0, start_idx - buffer_days)
            episodes.append(indices[buf_start: end_idx + 1])

    if not episodes:
        raise ValueError(
            f"No bear episodes found in training data with min_duration={min_duration_days} days"
        )

    # Combine all episode rows (deduplicated, sorted)
    all_dates = sorted(set(d for ep in episodes for d in ep))
    result = train.loc[all_dates]

    return result


def get_historical_stats(df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Calculate full-history performance statistics from the loaded DataFrame.
    Used in Step 1 to anchor the financial goal to real data.

    Args:
        df: DataFrame with at minimum a 'Close' column and DatetimeIndex.
        risk_free_rate: Annualised risk-free rate for Sharpe calculation.
            Defaults to RISK_FREE_RATE (India 10Y).  Pass the instrument-specific
            rate from instruments.json for non-INR instruments.
    """
    daily_returns = df["Close"].pct_change().dropna()
    years = len(df) / 252.0

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = (end_price / start_price) - 1
    cagr = (end_price / start_price) ** (1 / years) - 1

    annual_vol = daily_returns.std() * math.sqrt(252)

    sharpe = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std() * math.sqrt(252)

    # Rolling max drawdown
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Best and worst annual returns
    annual_rets = df["Close"].resample("YE").last().pct_change().dropna()

    # Last 12 months return (~252 trading days)
    lookback = min(252, len(df) - 1)
    last_12m_return = float((df["Close"].iloc[-1] / df["Close"].iloc[-lookback]) - 1)

    # Yearly returns for last 15 years (for chart)
    yearly_rets_15 = annual_rets.tail(15)
    yearly_returns_list = [
        {"year": str(idx.year), "return_pct": round(float(v * 100), 2)}
        for idx, v in yearly_rets_15.items()
    ]

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
        "last_12m_return_pct": round(last_12m_return * 100, 2),
        "yearly_returns": yearly_returns_list,
    }
