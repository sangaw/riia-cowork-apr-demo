"""
Portfolio Manager — FnO position parsing, Greeks, Margin, Stress, Payoff.

Data sources (all in rita_input/):
  positions-<date>.csv   — NSE positions export (latest file is used)
  orders-<date>.csv      — NSE orders export (latest file is used, not used yet)
  closed_positions.csv   — manually tracked legacy closed trades
  nifty_manual.csv       — NIFTY OHLCV (latest row = current market data)
  banknifty_manual.csv   — BANKNIFTY OHLCV (latest row)
  scenario_levels.csv    — user-defined Bull/Bear SL & Target per underlying

Lot sizes (update when NSE revises):
  NIFTY = 75, BANKNIFTY = 30
  NOTE: qty in NSE CSV is already in units (shares/contracts), not lots.
"""

import math
import os
import re
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from scipy.stats import norm

# ─── Constants ────────────────────────────────────────────────────────────────

LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30}

# SPAN rates (fraction of notional) for margin estimation
# Long options need no margin — only premium
SPAN_RATE = {
    "FUT": 0.10,   # ~10% of notional
    "CE_SHORT": 0.04,
    "PE_SHORT": 0.04,
}
EXPOSURE_RATE = 0.20   # Exposure = 20% of SPAN

RISK_FREE_RATE = 0.065  # 6.5% annualised

# Months excluded from ACTIVE positions display (nearly-expired)
EXCLUDE_EXPIRY_MONTHS = {"MAR"}

# Stress scenario moves (as fractions)
STRESS_MOVES = [-0.04, -0.02, 0.0, 0.02, 0.04]

# NIFTY price range for payoff chart
PAYOFF_POINTS = 15
PAYOFF_RANGE_PCT = 0.08   # ±8% from spot

# ─── Instrument name parser ───────────────────────────────────────────────────

_INST_RE = re.compile(
    r"^(NIFTY|BANKNIFTY)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d+)?(CE|PE|FUT)$"
)

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _last_thursday(year: int, month: int) -> date:
    """Return the last Thursday of the given month (NSE expiry)."""
    # Start from last day, walk back to Thursday
    last_day = date(year, month, 28)
    # Extend to actual last day
    next_month = month % 12 + 1
    next_year = year + (1 if month == 12 else 0)
    last_day = date(next_year, next_month, 1) - timedelta(days=1)
    # Find last Thursday (weekday 3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


def parse_instrument(raw: str) -> Optional[dict]:
    """
    Parse NSE instrument name into structured fields.

    Returns dict with: underlying, year_2d, exp_month, exp_date (date), type, strike
    Returns None if the name doesn't match.
    """
    m = _INST_RE.match(raw.strip())
    if not m:
        return None
    underlying, yr2, month_str, strike_str, inst_type = m.groups()
    yr_full = 2000 + int(yr2)
    month_num = _MONTH_MAP[month_str]
    exp_date = _last_thursday(yr_full, month_num)
    strike = int(strike_str) if strike_str else None

    # Human-readable display name
    if inst_type == "FUT":
        display = f"{underlying} {month_str} FUT"
    else:
        k = f"{strike:,}" if strike else "—"
        display = f"{underlying} {k} {inst_type}"

    return {
        "underlying": underlying,
        "year_2d": yr2,
        "exp_month": month_str,
        "exp_date": exp_date,
        "exp_label": f"{exp_date.day:02d}-{month_str[:1].upper()}{month_str[1:].lower()}-{exp_date.strftime('%y')}",
        "type": inst_type,
        "strike": strike,
        "display": display,
    }


# ─── Market data ─────────────────────────────────────────────────────────────

def load_market_data(input_dir: str) -> dict:
    """
    Load the latest row from nifty_manual.csv and banknifty_manual.csv.
    Returns { "NIFTY": {...}, "BANKNIFTY": {...} }
    """
    result = {}
    for und, fname in [("NIFTY", "nifty_manual.csv"), ("BANKNIFTY", "banknifty_manual.csv")]:
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            result[und] = None
            continue
        df = pd.read_csv(path).dropna(how="all")
        if df.empty:
            result[und] = None
            continue
        row      = df.iloc[-1]   # latest row
        prev_row = df.iloc[-2] if len(df) >= 2 else None

        def _parse_date(d) -> str:
            for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d-%b-%y"):
                try:
                    return datetime.strptime(str(d).strip(), fmt).strftime("%-d %b %Y")
                except ValueError:
                    pass
            return str(d)

        def _fmt_vol(v) -> str:
            try:
                return f"{int(float(str(v).replace(',', ''))):,}"
            except Exception:
                return str(v)

        vol_str = _fmt_vol(row.get("Shares Traded", row.get("Volume", 0)))
        close   = float(str(row.get("Close", 0)).replace(",", ""))
        open_   = float(str(row.get("Open",  0)).replace(",", ""))

        prev_close = (
            float(str(prev_row.get("Close", 0)).replace(",", ""))
            if prev_row is not None else None
        )

        result[und] = {
            "date":      _parse_date(row.get("Date", "")),
            "open":      open_,
            "high":      float(str(row.get("High", 0)).replace(",", "")),
            "low":       float(str(row.get("Low",  0)).replace(",", "")),
            "close":     close,
            "prevClose": prev_close,
            "volume":    vol_str,
            "shares":    vol_str,    # JS alias used in renderMarketSnapshot
            "turnover":  str(row.get("Turnover (₹ Cr)", row.get("Turnover", ""))),
        }
        result[und]["chgFromOpen"] = round(
            (close - open_) / open_ * 100, 2
        ) if open_ else 0.0
        result[und]["chgFromPrev"] = round(
            (close - prev_close) / prev_close * 100, 2
        ) if prev_close else None

    return result


def load_price_history(input_dir: str) -> list:
    """
    Return all rows from nifty_manual.csv and banknifty_manual.csv merged by date,
    augmented with per-day P&L totals from positions-<date>.csv files.

    Returns list of { date, nifty, banknifty, niftyPnl, bnknPnl } sorted oldest-first.
    """
    import re as _re

    rows_by_date: dict = {}

    # ── Step 1: build price rows from manual CSVs ─────────────────────────────
    for und, fname, key in [
        ("NIFTY",     "nifty_manual.csv",     "nifty"),
        ("BANKNIFTY", "banknifty_manual.csv",  "banknifty"),
    ]:
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).dropna(how="all")
        if df.empty:
            continue
        for _, row in df.iterrows():
            raw_date = str(row.get("Date", "")).strip()
            parsed = None
            for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d-%b-%y"):
                try:
                    parsed = datetime.strptime(raw_date, fmt)
                    break
                except ValueError:
                    pass
            if parsed is None:
                continue
            date_key = parsed.strftime("%d-%b-%Y").upper()   # e.g. "27-MAR-2026"
            close_val = float(str(row.get("Close", 0)).replace(",", ""))
            if date_key not in rows_by_date:
                rows_by_date[date_key] = {"date": date_key}
            rows_by_date[date_key][key] = close_val

    # ── Step 2: add P&L from positions-<date>.csv files ───────────────────────
    _MONTH_ABBR = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    pos_files = [
        f for f in os.listdir(input_dir)
        if f.lower().startswith("positions-") and f.lower().endswith(".csv")
    ]
    for fname in pos_files:
        # Parse DD + MMM from filename e.g. "positions-27mar.csv"
        m = _re.match(r"positions-(\d{1,2})([a-zA-Z]{3})\.csv", fname, _re.IGNORECASE)
        if not m:
            continue
        day, mon = int(m.group(1)), m.group(2).lower()
        month_num = _MONTH_ABBR.get(mon)
        if not month_num:
            continue

        # Match against a known date_key to get the year
        date_key = None
        for dk in rows_by_date:
            try:
                d = datetime.strptime(dk, "%d-%b-%Y")
                if d.day == day and d.month == month_num:
                    date_key = dk
                    break
            except ValueError:
                continue
        if date_key is None:
            continue

        # Parse P&L from positions file — active (unrealized) and closed (realized) separately
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_csv(fpath, quotechar='"')
            df.columns = [c.strip().strip('"') for c in df.columns]
            nifty_pnl = 0.0;  bnkn_pnl  = 0.0
            nifty_rpnl = 0.0; bnkn_rpnl = 0.0
            for _, row in df.iterrows():
                qty  = float(str(row.get("Qty.", 0)).replace(",", ""))
                inst = str(row.get("Instrument", "")).strip().strip('"')
                parsed = parse_instrument(inst)
                if not parsed:
                    continue
                if parsed["exp_month"] in EXCLUDE_EXPIRY_MONTHS:
                    continue
                pnl = float(str(row.get("P&L", 0)).replace(",", ""))
                und = parsed["underlying"]
                if qty == 0:
                    # Closed position — contributes to realized P&L
                    if und == "BANKNIFTY": bnkn_rpnl += pnl
                    elif und == "NIFTY":   nifty_rpnl += pnl
                else:
                    # Active — contributes to unrealized P&L (matches saveToday)
                    if und == "BANKNIFTY": bnkn_pnl += pnl
                    elif und == "NIFTY":   nifty_pnl += pnl
            rows_by_date[date_key]["niftyPnl"]  = round(nifty_pnl,  2)
            rows_by_date[date_key]["bnknPnl"]   = round(bnkn_pnl,   2)
            rows_by_date[date_key]["niftyRPnl"] = round(nifty_rpnl, 2)
            rows_by_date[date_key]["bnknRPnl"]  = round(bnkn_rpnl,  2)
        except Exception:
            pass

    complete = [
        v for v in rows_by_date.values()
        if v.get("nifty") and v.get("banknifty")
    ]
    complete.sort(key=lambda r: datetime.strptime(r["date"], "%d-%b-%Y"))
    return complete


def compute_orders_charges(input_dir: str) -> list:
    """
    Parse all orders-*.csv files and compute Zerodha F&O transaction charges per day.

    Zerodha F&O rates applied:
      Brokerage : ₹20 flat per COMPLETE order
      STT       : 0.1% of premium on options sell; 0.02% of notional on futures sell
      Exchange  : 0.035% of premium (options, both sides); 0.00173% of notional (futures, both sides)
      SEBI      : ₹10 per crore on total turnover (both sides)
      Stamp     : 0.003% on options buy; 0.002% on futures buy
      GST       : 18% on (brokerage + exchange + SEBI)

    Returns list of { date, brokerage, stt, exchange, sebi, stamp, gst, total } sorted oldest-first.
    """
    results = []
    ord_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().startswith("orders-") and f.lower().endswith(".csv")
    )

    for fname in ord_files:
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_csv(fpath, quotechar='"')
            df.columns = [c.strip().strip('"') for c in df.columns]

            status_col = next((c for c in df.columns if c.lower() == "status"), None)
            time_col   = next((c for c in df.columns if c.lower() == "time"), None)
            type_col   = next((c for c in df.columns if c.lower() == "type"), None)
            inst_col   = next((c for c in df.columns if c.lower() == "instrument"), None)
            qty_col    = next((c for c in df.columns if "qty" in c.lower()), None)
            price_col  = next((c for c in df.columns
                               if "avg" in c.lower() and "price" in c.lower()), None)

            if not all([status_col, time_col, type_col, inst_col, qty_col, price_col]):
                continue

            complete = df[df[status_col].str.strip().str.upper() == "COMPLETE"].copy()
            if complete.empty:
                continue

            # Derive trade date from first row's Time column ("2026-04-20 15:16:22")
            first_time = str(complete.iloc[0][time_col]).strip()
            try:
                trade_date = datetime.strptime(first_time[:10], "%Y-%m-%d")
            except ValueError:
                continue
            date_str = trade_date.strftime("%d-%b-%Y").upper()

            brokerage = 0.0
            stt       = 0.0
            exchange  = 0.0
            sebi      = 0.0
            stamp     = 0.0

            for _, row in complete.iterrows():
                order_type  = str(row[type_col]).strip().upper()
                instrument  = str(row[inst_col]).strip().upper()
                qty_str     = str(row[qty_col]).strip()
                filled_qty  = float(qty_str.split("/")[0]) if "/" in qty_str else float(qty_str or 0)
                avg_price   = float(str(row[price_col]).replace(",", "") or 0)

                if filled_qty <= 0 or avg_price <= 0:
                    continue

                is_option = instrument.endswith("CE") or instrument.endswith("PE")
                is_future = instrument.endswith("FUT")
                turnover  = filled_qty * avg_price

                brokerage += 20.0                          # ₹20 flat per order

                if is_option:
                    exchange += turnover * 0.00035         # 0.035% both sides
                    sebi     += turnover * 0.000001        # ₹10/Cr both sides
                    if order_type == "SELL":
                        stt  += turnover * 0.001           # 0.1% options sell
                    else:
                        stamp += turnover * 0.00003        # 0.003% options buy
                elif is_future:
                    exchange += turnover * 0.0000173       # 0.00173% both sides
                    sebi     += turnover * 0.000001        # ₹10/Cr both sides
                    if order_type == "SELL":
                        stt  += turnover * 0.0002          # 0.02% futures sell
                    else:
                        stamp += turnover * 0.00002        # 0.002% futures buy

            gst   = (brokerage + exchange + sebi) * 0.18
            total = brokerage + stt + exchange + sebi + stamp + gst

            results.append({
                "date":      date_str,
                "brokerage": round(brokerage, 2),
                "stt":       round(stt, 2),
                "exchange":  round(exchange, 2),
                "sebi":      round(sebi, 2),
                "stamp":     round(stamp, 2),
                "gst":       round(gst, 2),
                "total":     round(total, 2),
            })
        except Exception:
            continue

    results.sort(key=lambda r: datetime.strptime(r["date"], "%d-%b-%Y"))
    return results


# ─── Positions parser ────────────────────────────────────────────────────────

def _find_latest_csv(input_dir: str, prefix: str) -> Optional[str]:
    """Return most-recently-modified CSV matching prefix-*.csv in input_dir."""
    candidates = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().startswith(prefix) and f.lower().endswith(".csv")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_positions(input_dir: str) -> tuple[list, list]:
    """
    Parse the latest positions-*.csv.

    Returns:
        active   — list of position dicts (qty != 0, non-excluded expiry)
        closed   — list of closed position dicts (qty == 0, non-excluded expiry) + legacy CSV
    """
    path = _find_latest_csv(input_dir, "positions-")
    if not path:
        return [], []

    df = pd.read_csv(path, quotechar='"')
    # Normalise column names (NSE CSV has trailing empty column)
    df.columns = [c.strip().strip('"') for c in df.columns]

    active, closed = [], []

    for _, row in df.iterrows():
        raw_inst = str(row.get("Instrument", "")).strip().strip('"')
        parsed = parse_instrument(raw_inst)
        if not parsed:
            continue

        qty_raw  = float(str(row.get("Qty.", 0)).replace(",", ""))
        avg_raw  = float(str(row.get("Avg.", 0)).replace(",", ""))
        ltp_raw  = float(str(row.get("LTP", 0)).replace(",", ""))
        pnl_raw  = float(str(row.get("P&L", 0)).replace(",", ""))
        chg_raw  = float(str(row.get("Chg.", 0)).replace(",", ""))

        side = "Long" if qty_raw > 0 else "Short"
        qty_abs = abs(int(qty_raw))
        strike_fmt = f"{parsed['strike']:,}" if parsed["strike"] else "—"

        rec = {
            "instrument": raw_inst,
            "full":       parsed["display"],
            "underlying": parsed["underlying"],
            "und":        parsed["underlying"],   # JS alias
            "exp":        parsed["exp_month"],
            "expDate":    parsed["exp_label"],
            "exp_date":   parsed["exp_date"].isoformat(),
            "type":       parsed["type"],
            "strike":     strike_fmt,
            "strike_val": parsed["strike"],
            "side":       side,
            "signed_qty": int(qty_raw),
            "qty":        qty_abs,
            "avg":        avg_raw,
            "ltp":        ltp_raw,
            "pnl":        pnl_raw,
            "chg":        chg_raw,
        }

        if qty_abs == 0:
            # Closed position — skip MAR expiry (intraday managed separately)
            if parsed["exp_month"] not in EXCLUDE_EXPIRY_MONTHS:
                closed.append(rec)
        else:
            # Active — skip excluded expiry months
            if parsed["exp_month"] not in EXCLUDE_EXPIRY_MONTHS:
                active.append(rec)

    # Append legacy closed trades from closed_positions.csv
    legacy_path = os.path.join(input_dir, "closed_positions.csv")
    if os.path.exists(legacy_path):
        ldf = pd.read_csv(legacy_path)
        for _, row in ldf.iterrows():
            closed.append({
                "instrument": str(row.get("instrument", "")),
                "full":       str(row.get("instrument", "")),
                "underlying": str(row.get("underlying", "")),
                "exp":        str(row.get("expiry", "")).split("-")[1][:3].upper() if "-" in str(row.get("expiry","")) else "",
                "expDate":    str(row.get("expiry", "")),
                "exp_date":   "",
                "type":       str(row.get("type", "")),
                "strike":     str(row.get("strike", "")),
                "strike_val": None,
                "side":       str(row.get("side", "—")),
                "signed_qty": 0,
                "qty":        0,
                "avg":        0.0,
                "ltp":        0.0,
                "pnl":        float(row.get("pnl", 0)),
                "chg":        0.0,
            })

    return active, closed


# ─── Black-Scholes Greeks ─────────────────────────────────────────────────────

def _bs_d1d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def _bs_price(S, K, T, r, sigma, opt_type):
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0, S - K) if opt_type == "CE" else max(0, K - S)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(S, K, T, r, market_price, opt_type, tol=1e-5, max_iter=100) -> float:
    """Bisection IV solver. Returns default 0.15 if market price implies no solution."""
    if T <= 0 or market_price <= 0:
        return 0.15
    intrinsic = max(0, S - K) if opt_type == "CE" else max(0, K - S)
    if market_price < intrinsic:
        return 0.15

    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = _bs_price(S, K, T, r, mid, opt_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_greeks(active_positions: list, spot: dict, as_of: Optional[date] = None) -> list:
    """
    Compute approximate Black-Scholes greeks for each active position.

    Returns list of dicts with: full, und, exp, type, side, delta, theta, vega, iv
    Net delta for FUT = signed_qty (no BS needed).
    """
    today = as_of or date.today()
    result = []

    for p in active_positions:
        und  = p["underlying"]
        S    = spot.get(und, {}).get("close", 0) if spot.get(und) else 0
        if S <= 0:
            continue

        inst_type   = p["type"]
        signed_qty  = p["signed_qty"]  # positive=long, negative=short
        exp_date_str = p["exp_date"]

        if inst_type == "FUT":
            result.append({
                "full":    p["full"],
                "und":     und,
                "exp":     p["exp"],
                "type":    "FUT",
                "side":    p["side"],
                "delta":   signed_qty,
                "gamma":   0,
                "theta":   0,
                "vega":    0,
                "iv":      "—",
            })
            continue

        K = p["strike_val"]
        if not K or not exp_date_str:
            continue

        try:
            exp_d = date.fromisoformat(exp_date_str)
        except ValueError:
            continue

        T = max((exp_d - today).days / 365.0, 1 / 365)

        # IV from LTP
        ltp = p["ltp"]
        iv = _implied_vol(S, K, T, RISK_FREE_RATE, ltp, inst_type) if ltp > 0.5 else 0.15

        d1, d2 = _bs_d1d2(S, K, T, RISK_FREE_RATE, iv)
        if d1 is None:
            continue

        # Delta: N(d1) for CE, N(d1)-1 for PE — then multiply by signed_qty
        bs_delta = norm.cdf(d1) if inst_type == "CE" else norm.cdf(d1) - 1
        net_delta = round(bs_delta * signed_qty)

        nd1_pdf = norm.pdf(d1)

        # Gamma: N'(d1) / (S * sigma * sqrt(T)) × signed_qty  [per-point, scaled]
        gamma_raw = float(nd1_pdf) / (S * iv * math.sqrt(T))
        net_gamma = round(gamma_raw * signed_qty, 5)

        # Theta (per day): -(S*N'(d1)*sigma)/(2*sqrt(T)) ± r*K*e^{-rT}*N(±d2)
        if inst_type == "CE":
            theta_annual = (
                -(S * nd1_pdf * iv) / (2 * math.sqrt(T))
                - RISK_FREE_RATE * K * math.exp(-RISK_FREE_RATE * T) * norm.cdf(d2)
            )
        else:
            theta_annual = (
                -(S * nd1_pdf * iv) / (2 * math.sqrt(T))
                + RISK_FREE_RATE * K * math.exp(-RISK_FREE_RATE * T) * norm.cdf(-d2)
            )
        theta_per_day = round(theta_annual / 365 * signed_qty)

        # Vega: S * N'(d1) * sqrt(T) * 0.01  (₹ per 1% IV move) × signed_qty
        vega = round(S * nd1_pdf * math.sqrt(T) * 0.01 * signed_qty)

        result.append({
            "full":    p["full"],
            "und":     und,
            "exp":     p["exp"],
            "type":    inst_type,
            "side":    p["side"],
            "delta":   net_delta,
            "gamma":   net_gamma,
            "theta":   theta_per_day,
            "vega":    vega,
            "iv":      f"{iv*100:.1f}%",
        })

    return result


# ─── Net delta ───────────────────────────────────────────────────────────────

def compute_net_delta(greeks: list) -> dict:
    """Sum delta per underlying. Returns {"NIFTY": int, "BANKNIFTY": int}."""
    totals: dict = {}
    for g in greeks:
        und = g["und"]
        totals[und] = totals.get(und, 0) + g["delta"]
    return {k: round(v) for k, v in totals.items()}


def compute_net_greeks(greeks: list) -> dict:
    """
    Aggregate Δ, Γ, Θ, V per underlying for the summary cards.
    Returns { "NIFTY": {delta, gamma, theta, vega}, "BANKNIFTY": {...} }
    """
    agg: dict = {}
    for g in greeks:
        und = g["und"]
        if und not in agg:
            agg[und] = {"delta": 0, "gamma": 0.0, "theta": 0, "vega": 0}
        agg[und]["delta"] += g.get("delta", 0)
        agg[und]["gamma"] += g.get("gamma", 0)
        agg[und]["theta"] += g.get("theta", 0)
        agg[und]["vega"]  += g.get("vega",  0)
    # Round
    for und in agg:
        agg[und]["delta"] = round(int(agg[und]["delta"]))
        agg[und]["gamma"] = round(float(agg[und]["gamma"]), 4)
        agg[und]["theta"] = round(int(agg[und]["theta"]))
        agg[und]["vega"]  = round(int(agg[und]["vega"]))
    return agg


# ─── Margin estimation ───────────────────────────────────────────────────────

def compute_margin(active_positions: list, spot: dict) -> dict:
    """
    Estimate SPAN + Exposure margin per position.
    Long options have no margin (just premium).

    Returns:
        by_position — list of per-position margin dicts
        summary     — {und: {span, exposure, total}, "ALL": {...}}
    """
    by_pos = []
    summary: dict = {}

    for p in active_positions:
        und = p["underlying"]
        S   = spot.get(und, {}).get("close", 0) if spot.get(und) else 0
        if S <= 0:
            continue

        inst_type  = p["type"]
        side       = p["side"]
        qty        = p["qty"]

        if inst_type == "FUT":
            notional = qty * S
            span     = round(notional * SPAN_RATE["FUT"])
            exposure = round(span * EXPOSURE_RATE)
        elif side == "Short":
            notional = qty * S
            key      = f"{inst_type}_SHORT"
            span     = round(notional * SPAN_RATE.get(key, 0.04))
            exposure = round(span * EXPOSURE_RATE)
        else:
            # Long option — no margin
            span = 0
            exposure = 0

        total = span + exposure

        by_pos.append({
            "full":     p["full"],
            "und":      und,
            "exp":      p["exp"],
            "type":     inst_type,
            "side":     side,
            "qty":      qty,
            "span":     span,
            "exposure": exposure,
            "total":    total,
        })

        if und not in summary:
            summary[und] = {"span": 0, "exposure": 0, "total": 0}
        summary[und]["span"]     += span
        summary[und]["exposure"] += exposure
        summary[und]["total"]    += total

    # ALL aggregate
    summary["ALL"] = {
        "span":     sum(v["span"]     for v in summary.values()),
        "exposure": sum(v["exposure"] for v in summary.values()),
        "total":    sum(v["total"]    for v in summary.values()),
    }

    return {"by_position": by_pos, "summary": summary}


# ─── Stress scenarios ────────────────────────────────────────────────────────

def compute_stress(greeks: list, spot: dict, moves: list = STRESS_MOVES) -> list:
    """
    Delta-based stress scenarios for combined portfolio.
    Applies the same % move to all underlyings simultaneously.
    """
    scenarios = []
    for move in moves:
        total_pnl = 0
        per_und = {}
        for g in greeks:
            und = g["und"]
            S   = spot.get(und, {}).get("close", 0) if spot.get(und) else 0
            price_move = S * move
            pnl = g["delta"] * price_move
            total_pnl += pnl
            per_und[und] = per_und.get(und, 0) + pnl

        nifty_spot = spot.get("NIFTY", {}).get("close", 0) if spot.get("NIFTY") else 0
        scenarios.append({
            "move_pct":    move * 100,
            "move_label":  f"{'+' if move > 0 else ''}{move*100:.0f}%",
            "nifty_level": round(nifty_spot * (1 + move)),
            "pnl":         round(total_pnl),
            "per_und":     {k: round(v) for k, v in per_und.items()},
        })
    return scenarios


# ─── Payoff chart ─────────────────────────────────────────────────────────────

def compute_payoff(active_positions: list, spot: dict, underlying: str = "NIFTY",
                   as_of: Optional[date] = None) -> dict:
    """
    Option payoff at expiry for positions of a given underlying across a price range.
    Uses intrinsic value (ignores time value for simplicity).
    """
    und_spot = spot.get(underlying, {}).get("close", 0) if spot.get(underlying) else 0
    if und_spot <= 0:
        return {"labels": [], "data": []}

    lo = round(und_spot * (1 - PAYOFF_RANGE_PCT / 2), -2)
    hi = round(und_spot * (1 + PAYOFF_RANGE_PCT / 2), -2)
    step = round((hi - lo) / PAYOFF_POINTS, -2)
    if step <= 0:
        step = 500

    prices = [round(lo + i * step) for i in range(PAYOFF_POINTS + 1)]
    und_positions = [p for p in active_positions if p["underlying"] == underlying]

    payoff_data = []
    for price in prices:
        total = 0.0
        for p in und_positions:
            signed_qty = p["signed_qty"]
            inst_type  = p["type"]
            K          = p["strike_val"]
            avg        = p["avg"]

            if inst_type == "FUT":
                total += signed_qty * (price - avg)
            elif inst_type == "CE" and K:
                intrinsic = max(0, price - K)
                total += signed_qty * (intrinsic - avg)
            elif inst_type == "PE" and K:
                intrinsic = max(0, K - price)
                total += signed_qty * (intrinsic - avg)

        payoff_data.append(round(total))

    return {
        "labels": [str(p) for p in prices],
        "data":   payoff_data,
    }


# ─── Scenario levels ─────────────────────────────────────────────────────────

def load_scenario_levels(input_dir: str) -> dict:
    """
    Load Bull/Bear SL & Target from scenario_levels.csv.

    Expected columns: underlying, mode, sl, target
    Returns nested dict: { "NIFTY": { "bull": {sl, target}, "bear": {...} }, ... }
    """
    path = os.path.join(input_dir, "scenario_levels.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    result: dict = {}
    for _, row in df.iterrows():
        und  = str(row["underlying"]).strip().upper()
        mode = str(row["mode"]).strip().lower()
        if und not in result:
            result[und] = {}
        result[und][mode] = {
            "sl":     int(row["sl"]),
            "target": int(row["target"]),
        }
    return result


def load_ledger_balance(input_dir: str) -> float:
    """Read ledger_balance from scenario_levels.csv if present, else default."""
    path = os.path.join(input_dir, "scenario_levels.csv")
    if not os.path.exists(path):
        return 3_500_000.0
    df = pd.read_csv(path)
    if "ledger_balance" in df.columns:
        val = df["ledger_balance"].dropna().iloc[0] if len(df) else None
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 3_500_000.0


# ─── Hedge Quality Scorer ─────────────────────────────────────────────────────

def _hqs_score(pct_otm: float, dte: int, delta_abs: float) -> int:
    """
    Hedge Quality Score (0–100) for a single long option position.
    Penalises: strike too far OTM, too few DTE, too low delta.
    """
    score = 100

    # Strike distance from spot
    if pct_otm > 15:
        score -= 45
    elif pct_otm > 10:
        score -= 35
    elif pct_otm > 7:
        score -= 25
    elif pct_otm > 5:
        score -= 15
    elif pct_otm > 3:
        score -= 5

    # Days to expiry — danger zone in last 2 weeks
    if dte < 7:
        score -= 35
    elif dte < 14:
        score -= 20
    elif dte < 21:
        score -= 10

    # Delta — probability of reaching the strike
    if delta_abs < 0.05:
        score -= 20
    elif delta_abs < 0.10:
        score -= 12
    elif delta_abs < 0.20:
        score -= 5

    return max(0, min(100, score))


def compute_hedge_quality(
    active_positions: list,
    spot: dict,
    as_of: Optional[date] = None,
) -> dict:
    """
    Score all long option positions on hedge quality (HQS 0–100).

    Tiers:
      green  (≥65) — Good Hedge: reasonable strike, DTE, delta
      yellow (40–64) — Watch: marginal on one or more dimensions
      red    (<40)   — Lottery Ticket: high decay risk, far OTM or near expiry

    Returns:
        positions — scored list sorted worst-first
        summary   — portfolio-level hedge health totals
    """
    today = as_of or date.today()
    scored = []

    for p in active_positions:
        if p["type"] == "FUT" or p["side"] != "Long" or not p["strike_val"]:
            continue

        und = p["underlying"]
        S = spot.get(und, {}).get("close", 0) if spot.get(und) else 0
        if S <= 0:
            continue

        K = p["strike_val"]
        try:
            exp_d = date.fromisoformat(p["exp_date"])
        except (ValueError, KeyError):
            continue

        dte = max(0, (exp_d - today).days)
        T = max(dte / 365.0, 1 / 365)

        # % distance from spot — CE: above; PE: below
        pct_otm = (
            max(0.0, (K - S) / S * 100) if p["type"] == "CE"
            else max(0.0, (S - K) / S * 100)
        )

        # Delta via Black-Scholes
        ltp = p["ltp"]
        iv = _implied_vol(S, K, T, RISK_FREE_RATE, ltp, p["type"]) if ltp > 0.5 else 0.15
        d1, _ = _bs_d1d2(S, K, T, RISK_FREE_RATE, iv)
        if d1 is not None:
            delta_abs = norm.cdf(d1) if p["type"] == "CE" else 1.0 - norm.cdf(d1)
        else:
            delta_abs = 0.01

        score = _hqs_score(pct_otm, dte, delta_abs)
        tier  = "green" if score >= 65 else "yellow" if score >= 40 else "red"
        label = {"green": "Good Hedge", "yellow": "Watch", "red": "Lottery Ticket"}[tier]
        premium_total = round(p["qty"] * p["avg"])

        scored.append({
            "instrument":    p["instrument"],
            "full":          p["full"],
            "und":           und,
            "exp":           p["exp"],
            "type":          p["type"],
            "strike":        p["strike"],
            "qty":           p["qty"],
            "avg":           p["avg"],
            "ltp":           p["ltp"],
            "pnl":           p["pnl"],
            "pct_otm":       round(pct_otm, 1),
            "dte":           dte,
            "delta_abs":     round(delta_abs, 3),
            "premium_total": premium_total,
            "hqs":           score,
            "hqs_tier":      tier,
            "hqs_label":     label,
        })

    # Worst first (red → yellow → green, within tier ascending score)
    tier_order = {"red": 0, "yellow": 1, "green": 2}
    scored.sort(key=lambda x: (tier_order[x["hqs_tier"]], x["hqs"]))

    total_premium = sum(s["premium_total"] for s in scored)
    red_premium   = sum(s["premium_total"] for s in scored if s["hqs_tier"] == "red")
    total_qty     = sum(s["qty"] for s in scored)
    avg_delta     = (
        float(round(sum(s["delta_abs"] * s["qty"] for s in scored) / total_qty, 3))
        if total_qty > 0 else 0.0
    )
    pnl_at_risk = sum(s["pnl"] for s in scored if s["hqs_tier"] == "red")

    return {
        "positions": scored,
        "summary": {
            "total_premium":   total_premium,
            "red_premium":     red_premium,
            "count_red":       sum(1 for s in scored if s["hqs_tier"] == "red"),
            "count_yellow":    sum(1 for s in scored if s["hqs_tier"] == "yellow"),
            "count_green":     sum(1 for s in scored if s["hqs_tier"] == "green"),
            "total_long_opts": len(scored),
            "avg_delta":       avg_delta,
            "pnl_at_risk":     round(pnl_at_risk),
        },
    }


# ─── Main summary builder ────────────────────────────────────────────────────

def get_portfolio_summary(input_dir: str) -> dict:
    """
    Build the complete portfolio summary dict for the API endpoint.
    All sections computed from CSVs — nothing hardcoded.
    """
    market   = load_market_data(input_dir)
    active, closed = parse_positions(input_dir)
    greeks     = compute_greeks(active, market)
    net_delta  = compute_net_delta(greeks)
    net_greeks = compute_net_greeks(greeks)
    margin_data = compute_margin(active, market)
    stress   = compute_stress(greeks, market)
    payoff   = {
        "NIFTY":     compute_payoff(active, market, "NIFTY"),
        "BANKNIFTY": compute_payoff(active, market, "BANKNIFTY"),
    }
    scenario_levels = load_scenario_levels(input_dir)
    ledger   = load_ledger_balance(input_dir)
    hedge_quality = compute_hedge_quality(active, market)

    # Realized P&L = sum of closed positions P&L
    realized_pnl = round(sum(p["pnl"] for p in closed), 2)

    # Margin utilization
    margin_summary = margin_data["summary"]
    util: dict = {}
    for und, vals in margin_summary.items():
        total = vals["total"]
        util[und] = round(total / ledger * 100, 1) if ledger > 0 else 0.0

    return {
        "as_of":           (market.get("NIFTY") or {}).get("date", ""),
        "market":          market,
        "positions":       active,
        "closed_positions": closed,
        "realized_pnl":    realized_pnl,
        "greeks":          greeks,
        "net_delta":       net_delta,
        "net_greeks":      net_greeks,
        "margin": {
            "by_position":  margin_data["by_position"],
            "summary":      margin_summary,
            "ledger":       ledger,
            "utilization":  util,
        },
        "stress":          stress,
        "payoff":          payoff,
        "scenario_levels": scenario_levels,
        "hedge_quality":   hedge_quality,
    }
