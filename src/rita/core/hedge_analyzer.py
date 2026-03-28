"""
Hedge History Analyzer — multi-day hedging pattern analysis.

Reads all positions-*.csv files to produce a temporal view of:
  - Daily hedge premium bucketed by OTM distance (near-ATM / mid-OTM / far-OTM)
  - Reactive vs proactive hedge additions (added on down days vs calm days)
  - Anchor positions — long options held 4+ days with declining quality
  - Hedge budget utilisation vs futures notional
"""

import os
import re
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from scipy.stats import norm

from rita.core.portfolio_manager import (
    RISK_FREE_RATE,
    _bs_d1d2,
    _hqs_score,
    _implied_vol,
    parse_instrument,
)

# ─── Constants ────────────────────────────────────────────────────────────────

_MONTH_ABB = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_YEAR = 2026          # current expiry year — extend as needed
_DOWN_THRESHOLD = -0.8  # % change below which a day is "down"
_ANCHOR_MIN_DAYS = 4    # min consecutive days to flag as anchor
_ANCHOR_MAX_HQS  = 60   # avg HQS below this = anchor


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _file_date(fname: str) -> Optional[date]:
    """Extract date from positions-DDmon.csv  (e.g. positions-16mar.csv → 2026-03-16)."""
    m = re.match(r"positions-(\d{1,2})([a-z]+)\.csv", fname.lower())
    if not m:
        return None
    day = int(m.group(1))
    mon = _MONTH_ABB.get(m.group(2)[:3])
    if not mon:
        return None
    try:
        return date(_YEAR, mon, day)
    except ValueError:
        return None


def _load_spot_series(input_dir: str) -> dict:
    """
    Load daily close prices from nifty_manual.csv and banknifty_manual.csv.
    Returns {date: {"NIFTY": float, "BANKNIFTY": float}}
    """
    series: dict = {}
    for und, fname in [("NIFTY", "nifty_manual.csv"), ("BANKNIFTY", "banknifty_manual.csv")]:
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            raw = str(row.get("Date", "")).strip().title()  # "16-MAR-2026" → "16-Mar-2026"
            close = float(str(row.get("Close", 0)).replace(",", ""))
            d: Optional[date] = None
            for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d-%b-%y"):
                try:
                    d = datetime.strptime(raw, fmt).date()
                    break
                except ValueError:
                    pass
            if d:
                if d not in series:
                    series[d] = {}
                series[d][und] = close
    return series


def _nearest_spot(series: dict, d: date, und: str) -> float:
    """Return close for `und` on `d` or nearest previous trading day (up to 6 days back)."""
    for offset in range(7):
        v = series.get(d - timedelta(days=offset), {}).get(und, 0)
        if v > 0:
            return float(v)
    return 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def compute_hedge_history(input_dir: str) -> dict:
    """
    Analyze hedging patterns across all positions-*.csv files.

    Returns
    -------
    days            — per-day summary dicts (for timeline chart & table)
    anchors         — long options held 4+ days with avg HQS < ANCHOR_MAX_HQS
    reactive_score  — {total_new_opts, reactive_opts, reactive_pct, label}
    budget          — {avg_pct, max_pct, max_date, days_over_5pct, recommended_max}
    """
    spot_series = _load_spot_series(input_dir)

    # Collect & sort positions files by date
    files: list = []
    for fname in os.listdir(input_dir):
        d = _file_date(fname)
        if d:
            files.append((d, os.path.join(input_dir, fname)))
    files.sort()

    if not files:
        return {"days": [], "anchors": [], "reactive_score": {}, "budget": {}}

    days_raw: list[dict] = []
    prev_instruments: set = set()
    inst_history: dict = {}  # instrument → {full, und, avg, pct_otm_at_entry, hqs_at_entry, entries:[]}

    for day_date, fpath in files:
        nifty_spot = _nearest_spot(spot_series, day_date, "NIFTY")
        bnkn_spot  = _nearest_spot(spot_series, day_date, "BANKNIFTY")

        # Previous trading day close for NIFTY change %
        prev_nifty = 0.0
        for offset in range(1, 7):
            prev_nifty = _nearest_spot(spot_series, day_date - timedelta(days=offset), "NIFTY")
            if prev_nifty > 0:
                break

        nifty_chg_pct = (
            round((nifty_spot - prev_nifty) / prev_nifty * 100, 2)
            if prev_nifty > 0 and nifty_spot > 0 else 0.0
        )

        try:
            df = pd.read_csv(fpath, quotechar='"')
            df.columns = [c.strip().strip('"') for c in df.columns]
        except Exception:
            continue

        long_opts: list[dict] = []
        futures_notional = 0.0
        current_instruments: set = set()

        for _, row in df.iterrows():
            raw_inst = str(row.get("Instrument", "")).strip().strip('"')
            parsed = parse_instrument(raw_inst)
            if not parsed:
                continue

            qty_raw = float(str(row.get("Qty.",  0)).replace(",", ""))
            if qty_raw == 0:
                continue

            avg_raw = float(str(row.get("Avg.",  0)).replace(",", ""))
            ltp_raw = float(str(row.get("LTP",   0)).replace(",", ""))
            pnl_raw = float(str(row.get("P&L",   0)).replace(",", ""))

            current_instruments.add(raw_inst)
            und  = parsed["underlying"]
            spot = nifty_spot if und == "NIFTY" else bnkn_spot

            # Futures — accumulate notional only
            if parsed["type"] == "FUT":
                futures_notional += abs(qty_raw) * (spot or 0)
                continue

            # Only score long options
            if qty_raw <= 0:
                continue

            K     = parsed["strike"] or 0
            exp_d = parsed["exp_date"]
            dte   = max(0, (exp_d - day_date).days)
            T     = max(dte / 365.0, 1 / 365)

            if K > 0 and spot > 0:
                if parsed["type"] == "CE":
                    pct_otm = max(0.0, (K - spot) / spot * 100)
                else:
                    pct_otm = max(0.0, (spot - K) / spot * 100)

                iv = (
                    _implied_vol(spot, K, T, RISK_FREE_RATE, ltp_raw, parsed["type"])
                    if ltp_raw > 0.5 else 0.15
                )
                d1, _ = _bs_d1d2(spot, K, T, RISK_FREE_RATE, iv)
                if d1 is not None:
                    delta_abs = (
                        float(norm.cdf(d1)) if parsed["type"] == "CE"
                        else float(1.0 - norm.cdf(d1))
                    )
                else:
                    delta_abs = 0.01

                hqs  = _hqs_score(pct_otm, dte, delta_abs)
                tier = "green" if hqs >= 65 else "yellow" if hqs >= 40 else "red"
            else:
                pct_otm   = 0.0
                delta_abs = 0.0
                hqs       = 50
                tier      = "yellow"

            premium = round(abs(qty_raw) * avg_raw)
            is_new  = raw_inst not in prev_instruments

            long_opts.append({
                "instrument": raw_inst,
                "full":       parsed["display"],
                "und":        und,
                "qty":        int(abs(qty_raw)),
                "avg":        avg_raw,
                "ltp":        ltp_raw,
                "pnl":        pnl_raw,
                "pct_otm":    round(pct_otm, 1),
                "dte":        dte,
                "delta_abs":  round(delta_abs, 3),
                "premium":    premium,
                "hqs":        hqs,
                "hqs_tier":   tier,
                "is_new":     is_new,
            })

            # Track per-instrument history
            if raw_inst not in inst_history:
                inst_history[raw_inst] = {
                    "full":             parsed["display"],
                    "und":              und,
                    "avg":              avg_raw,
                    "pct_otm_at_entry": round(pct_otm, 1),
                    "hqs_at_entry":     hqs,
                    "entries":          [],
                }
            inst_history[raw_inst]["entries"].append({
                "date":     day_date.strftime("%d-%b"),
                "ltp":      ltp_raw,
                "pnl":      pnl_raw,
                "hqs":      hqs,
                "hqs_tier": tier,
                "pct_otm":  round(pct_otm, 1),
            })

        # ── Bucket premiums ───────────────────────────────────────────────────
        near_atm = sum(p["premium"] for p in long_opts if p["pct_otm"] <= 5)
        mid_otm  = sum(p["premium"] for p in long_opts if 5  < p["pct_otm"] <= 10)
        far_otm  = sum(p["premium"] for p in long_opts if p["pct_otm"] > 10)
        total    = near_atm + mid_otm + far_otm

        new_count  = sum(1 for p in long_opts if p["is_new"])
        react_new  = sum(1 for p in long_opts if p["is_new"] and nifty_chg_pct <= _DOWN_THRESHOLD)
        budget_pct = round(total / futures_notional * 100, 1) if futures_notional > 0 else 0.0

        days_raw.append({
            "date_obj":           day_date,                    # stripped before return
            "date":               day_date.strftime("%d-%b"),
            "nifty_close":        nifty_spot,
            "nifty_chg_pct":      nifty_chg_pct,
            "total_premium":      total,
            "near_atm_premium":   near_atm,
            "mid_otm_premium":    mid_otm,
            "far_otm_premium":    far_otm,
            "futures_notional":   round(futures_notional),
            "hedge_budget_pct":   budget_pct,
            "new_opts_count":     new_count,
            "reactive_new_count": react_new,
            "is_down_day":        nifty_chg_pct <= _DOWN_THRESHOLD,
            "hqs_counts": {
                "green":  sum(1 for p in long_opts if p["hqs_tier"] == "green"),
                "yellow": sum(1 for p in long_opts if p["hqs_tier"] == "yellow"),
                "red":    sum(1 for p in long_opts if p["hqs_tier"] == "red"),
            },
        })
        prev_instruments = current_instruments

    # ── Anchor positions ──────────────────────────────────────────────────────
    anchors: list[dict] = []
    for inst, info in inst_history.items():
        n = len(info["entries"])
        if n < _ANCHOR_MIN_DAYS:
            continue
        avg_hqs = sum(e["hqs"] for e in info["entries"]) / n
        if avg_hqs >= _ANCHOR_MAX_HQS:
            continue  # consistently acceptable — not an anchor
        last  = info["entries"][-1]
        first = info["entries"][0]
        decay = (
            round((last["ltp"] - info["avg"]) / info["avg"] * 100, 1)
            if info["avg"] > 0 else 0.0
        )
        anchors.append({
            "instrument":        inst,
            "full":              info["full"],
            "und":               info["und"],
            "days_held":         n,
            "entry_avg":         info["avg"],
            "current_ltp":       last["ltp"],
            "pct_decay":         decay,
            "pct_otm_at_entry":  info["pct_otm_at_entry"],
            "hqs_first":         first["hqs"],
            "hqs_current":       last["hqs"],
            "current_pnl":       last["pnl"],
            "ltp_history":       [e["ltp"]  for e in info["entries"]],
            "date_history":      [e["date"] for e in info["entries"]],
        })
    anchors.sort(key=lambda x: x["pct_decay"])  # worst decay first

    # ── Reactive score ────────────────────────────────────────────────────────
    total_new = sum(d["new_opts_count"]     for d in days_raw)
    react_new = sum(d["reactive_new_count"] for d in days_raw)
    react_pct = round(react_new / total_new * 100) if total_new > 0 else 0
    react_label = (
        "Highly Reactive" if react_pct >= 60
        else "Moderately Reactive" if react_pct >= 30
        else "Proactive"
    )

    # ── Budget summary ────────────────────────────────────────────────────────
    bpcts   = [d["hedge_budget_pct"] for d in days_raw if d["hedge_budget_pct"] > 0]
    max_day = max(days_raw, key=lambda d: d["hedge_budget_pct"]) if days_raw else None

    days_out = [{k: v for k, v in d.items() if k != "date_obj"} for d in days_raw]

    return {
        "days": days_out,
        "anchors": anchors,
        "reactive_score": {
            "total_new_opts": total_new,
            "reactive_opts":  react_new,
            "reactive_pct":   react_pct,
            "label":          react_label,
        },
        "budget": {
            "avg_pct":         round(sum(bpcts) / len(bpcts), 1) if bpcts else 0,
            "max_pct":         max_day["hedge_budget_pct"] if max_day else 0,
            "max_date":        max_day["date"] if max_day else "",
            "days_over_5pct":  sum(1 for p in bpcts if p > 5.0),
            "recommended_max": 5.0,
        },
    }
