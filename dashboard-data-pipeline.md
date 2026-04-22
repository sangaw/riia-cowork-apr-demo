# Dashboard Data Pipeline — Instrument Onboarding

> How raw OHLCV data becomes a fully-featured CSV that powers the RITA dashboard and APIs.

---

## Architecture overview

```
Raw source CSV  →  load_nifty_csv()  →  calculate_indicators()  →  Fully-featured CSV
                                                                         ↓
                                                               rita_output/<id>_merged.csv
                                                                         ↓
                                                               load_prepared_csv()  (API startup)
                                                                         ↓
                                                               WorkflowOrchestrator._feat_df
                                                                         ↓
                                                          All endpoints: market signals, backtest, etc.
```

### Why bake indicators into the CSV?

| Approach | Accuracy | Performance | Portability |
|---|---|---|---|
| Compute in-memory on date slice | ❌ EMA warmup is short — indicators less accurate | OK | Poor |
| Compute on full history, store in CSV | ✅ Full 25-year warmup for all EMAs | Fast (read once) | Self-contained |

Computing indicators on the **full history first** then slicing is the correct pattern. A 200-day EMA computed on 50 rows is wrong; computed on 6,000 rows it is accurate.

---

## Instrument registry — `rita_input/instruments.json`

Each instrument entry:

```json
{
  "id": "nvda",
  "name": "NVIDIA",
  "exchange": "NASDAQ",
  "currency": "USD",
  "flag": "🇺🇸",
  "description": "NVIDIA Corporation — AI compute and GPU technology leader",
  "prepared_csv": "nvda_merged.csv",
  "color": "warn",
  "risk_free_rate": 0.043
}
```

| Field | Purpose |
|---|---|
| `id` | URL-safe identifier, used everywhere |
| `prepared_csv` | Filename expected in `rita_output/` |
| `color` | CSS variable name (`build`, `run`, `mon`, `warn`) |
| `risk_free_rate` | Annualised rate for Sharpe calculations (country-specific) |

### Risk-free rates by region

| Instrument | Currency | Rate | Benchmark |
|---|---|---|---|
| Nifty 50, Bank Nifty | INR | 0.07 | India 10Y Gsec |
| ASML | EUR | 0.025 | Germany 10Y Bund |
| NVIDIA | USD | 0.043 | US 10Y Treasury |

`data_loader.py` uses `RISK_FREE_RATE = 0.07` as the module-level default. Pass the instrument-specific rate into `get_historical_stats()` when available.

---

## Data loader — `src/rita/core/data_loader.py`

### `load_nifty_csv(csv_path)` — raw OHLCV loader

Handles two date formats:
- IST timezone-aware (`1999-07-01 00:00:00+05:30`) — Nifty NSE export
- Plain ISO (`2001-03-09`) — ASML, NVDA, yfinance output

Returns a DataFrame with `DatetimeIndex` and columns `Open, High, Low, Close, Volume`.

### `load_prepared_csv(csv_path)` — fully-featured loader

Added alongside `load_nifty_csv`. Detection logic:

```python
if "rsi_14" in df.columns:
    # Pre-computed — use as-is, restore DatetimeIndex, return
else:
    # OHLCV-only — delegate to load_nifty_csv() path
```

Used by `WorkflowOrchestrator.__init__` so that fully-featured CSVs skip recalculation.

---

## Technical indicators — `src/rita/core/technical_analyzer.py`

`calculate_indicators(df)` adds these columns to an OHLCV DataFrame:

| Column | Description |
|---|---|
| `rsi_14` | RSI 14-period |
| `macd` | MACD line (12/26) |
| `macd_signal` | MACD signal line (9) |
| `macd_hist` | MACD histogram |
| `bb_upper/mid/lower` | Bollinger Bands (20, 2σ) |
| `bb_pct_b` | %B position (0=lower, 1=upper) |
| `atr_14` | Average True Range 14-period |
| `ema_5/13/26/50/200` | Exponential Moving Averages |
| `trend_score` | Normalised slope of EMA-50 over 20 days, clipped to [−1, +1] |
| `ema_ratio` | EMA-26 / EMA-50 clipped to [0.5, 1.5] — regime signal (Feature 9) |
| `daily_return` | Close-to-close daily return |

`detect_regime(df)` uses `ema_ratio < 0.99` for ≥3 consecutive days → BEAR, else BULL.

---

## Workflow orchestrator — `src/rita/orchestration/workflow.py`

`WorkflowOrchestrator.__init__` startup:

```python
self._raw_df, self._feat_df = load_prepared_csv(self.csv_path)
# If CSV is fully-featured: _raw_df = OHLCV slice, _feat_df = full pre-computed DF
# If CSV is OHLCV-only:    _raw_df = loaded DF,    _feat_df = calculate_indicators(_raw_df)
```

When CSV is fully-featured, date slices for train/val/backtest are taken directly from `self._feat_df` — no `calculate_indicators()` re-call needed. The ~5 call sites in the workflow that previously called `calculate_indicators(get_xxx_data(self._raw_df))` now call `get_xxx_data(self._feat_df)` for fully-featured instruments.

---

## Onboarding a new instrument — step-by-step

### 1. Add to instrument registry

Edit `rita_input/instruments.json` — add entry with all fields including `risk_free_rate`.

### 2. Get the source data

**Historical CSV (yfinance):**
```python
import yfinance as yf
df = yf.download("NVDA", start="2000-01-01", auto_adjust=True)
df.to_csv("nvda_raw.csv")
```

**Or use an existing CSV** if already sourced externally (e.g. `nvda_daily_25yr_rounded.csv`).

Source CSV must have columns: `Date, Open, High, Low, Close, Volume` with dates as `YYYY-MM-DD`.

### 3. Run the preparation script

```bash
python scripts/prepare_nvda.py
# generically:
python scripts/prepare_instrument.py --id nvda --source path/to/source.csv
```

The script:
1. Loads source CSV via `load_nifty_csv()`
2. Runs `calculate_indicators()` on the full history
3. Saves all columns to `rita_output/<id>_merged.csv`

### 4. Verify

```bash
python -c "
import pandas as pd
df = pd.read_csv('rita_output/nvda_merged.csv')
print(df.shape)           # expect (rows, ~21 columns)
print(df.columns.tolist())
print(df.tail(2))
"
```

### 5. Restart the API

```powershell
python run_api.py
```

The instrument will now appear as `data_ready: true` in the instrument selector on the landing page (`/dashboard/`).

---

## Ongoing data refresh (NVDA / international instruments)

NSE instruments (Nifty, Bank Nifty) use the `/api/v1/prepare-data` endpoint with NSE-format CSVs.

International instruments use **yfinance**:

```python
import yfinance as yf, pandas as pd

# Fetch only new rows since last date in prepared CSV
existing = pd.read_csv("rita_output/nvda_merged.csv", parse_dates=["Date"])
last_date = existing["Date"].max()

new = yf.download("NVDA", start=last_date.strftime("%Y-%m-%d"), auto_adjust=True)
# ... append, recalculate indicators on full combined history, save
```

A `/api/v1/refresh-data` endpoint (planned) will wrap this logic.

---

## NVDA-specific notes

- **Source file:** `C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\raw-data\nvidia\nvda_daily_25yr_rounded.csv`
- **Date range:** 2001-04-09 to 2026-04-02 (6,283 rows)
- **Format:** Plain ISO dates, standard OHLCV, values already rounded — no cleaning needed
- **Risk-free rate:** 0.043 (US 10Y Treasury)
- **Model training:** Not yet done — train/val/backtest date splits TBD (different from Nifty's 2010/2023/2025 boundaries)
- **Data refresh:** yfinance `"NVDA"` ticker

---

## File locations

| File | Role |
|---|---|
| `rita_input/instruments.json` | Instrument registry |
| `rita_output/<id>_merged.csv` | Fully-featured prepared CSV (OHLCV + indicators) |
| `scripts/prepare_nvda.py` | One-time preparation script for NVDA |
| `src/rita/core/data_loader.py` | `load_nifty_csv()`, `load_prepared_csv()` |
| `src/rita/core/technical_analyzer.py` | `calculate_indicators()` |
| `src/rita/orchestration/workflow.py` | `WorkflowOrchestrator` — consumes prepared CSV |
| `src/rita/interfaces/rest_api.py` | `_get_active_csv()`, `/api/v1/instruments` |
