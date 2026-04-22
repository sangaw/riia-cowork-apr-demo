"""
Prepare Bank Nifty fully-featured CSV for the RITA dashboard.

Loads the raw Bank Nifty OHLCV CSV, computes all technical indicators on
the full history, and writes the combined result to rita_output/banknifty_merged.csv.

Usage:
    python scripts/prepare_banknifty.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rita.core.data_loader import load_nifty_csv
from rita.core.technical_analyzer import calculate_indicators
from rita.config import OUTPUT_DIR

SOURCE_CSV = (
    r"C:\Users\Sandeep\Documents\Work\code\poc\alphavantage-api-demo"
    r"\banknifty_daily_25yr_rounded.csv"
)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "banknifty_merged.csv")


def main():
    print(f"Loading source: {SOURCE_CSV}")
    raw = load_nifty_csv(SOURCE_CSV)
    print(f"  Rows loaded : {len(raw):,}  ({raw.index[0].date()} to {raw.index[-1].date()})")

    print("Computing technical indicators on full history...")
    feat = calculate_indicators(raw)

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
