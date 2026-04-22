"""
Prepare NVDA fully-featured CSV for the RITA dashboard.

Loads the raw 25-year NVDA OHLCV CSV, computes all technical indicators on
the full history, and writes the combined result to rita_output/nvda_merged.csv.

Usage:
    python scripts/prepare_nvda.py

The output file is detected by load_prepared_csv() (rsi_14 column present) and
used directly by the WorkflowOrchestrator without re-running calculate_indicators().
"""

import os
import sys

# Allow running from repo root or from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rita.core.data_loader import load_nifty_csv
from rita.core.technical_analyzer import calculate_indicators
from rita.config import OUTPUT_DIR

SOURCE_CSV = (
    r"C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading"
    r"\data\raw-data\nvidia\nvda_daily_25yr_rounded.csv"
)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "nvda_merged.csv")


def main():
    print(f"Loading source: {SOURCE_CSV}")
    raw = load_nifty_csv(SOURCE_CSV)
    print(f"  Rows loaded : {len(raw):,}  ({raw.index[0].date()} to {raw.index[-1].date()})")

    print("Computing technical indicators on full history…")
    feat = calculate_indicators(raw)

    # Persist with Date as a plain column (consistent with nifty_merged.csv convention)
    out = feat.copy()
    out.index.name = "Date"
    out = out.reset_index()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    cols = out.columns.tolist()
    print(f"  Columns ({len(cols)}): {cols}")
    print(f"  Rows written: {len(out):,}")
    print(f"  Output      : {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
