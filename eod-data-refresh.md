# EOD Data Refresh — RITA

Run these steps after market close each trading day.

---

## 1. Append NIFTY price row to `nifty_merged.csv`

**Source:** NSE website → NIFTY 50 historical data (today's row)

Open `rita_output/nifty_merged.csv` and append one row at the bottom:

```
2026-04-02 00:00:00+05:30,22383.40,22782.30,22182.55,22713.10,495121035,38669.56
```

**Format rules:**
- Date: `YYYY-MM-DD 00:00:00+05:30`
- Columns (in order): `date,open,high,low,close,shares traded,turnover (₹ cr)`
- Numbers: plain decimals — **no Indian comma formatting** (fix `22,824.35` → `22824.35` before pasting)
- Turnover is in ₹ Cr

> **Do NOT run `prepare_data()` for this.** Direct append is the safe method. Running prepare_data while `banknifty_manual.csv` is in `rita_input/` will mix BANKNIFTY rows into the NIFTY series.

---

## 2. Update `rita_input/nifty_manual.csv`

This file drives the FnO dashboard's current NIFTY price and the P&L history chart.

Add today's row at the bottom in NSE export format:

```
02-APR-2026,22383.40,22782.30,22182.55,22713.10,495121035,38669.56
```

**Format rules:**
- Date: `DD-MMM-YYYY` (uppercase month, e.g. `APR`)
- Same column order: `Date,Open,High,Low,Close,Shares Traded,Turnover (₹ Cr)`
- Plain decimals — no Indian comma formatting

---

## 3. Update `rita_input/banknifty_manual.csv`

Same as above, for BANKNIFTY. Add today's row:

```
02-APR-2026,50625.65,51731.95,49954.85,51548.75,339187207,12190.88
```

Same format: `DD-MMM-YYYY`, plain decimals.

---

## 4. Save today's positions export

**Source:** Zerodha Kite → Positions → Export CSV

Save the file as `rita_input/positions-DDmmm.csv`:

- Naming pattern: `positions-02apr.csv` (day + 3-letter lowercase month)
- The dashboard always picks the most-recently-modified file matching `positions-*.csv`

---

## 5. Save today's orders export (optional)

**Source:** Zerodha Kite → Orders → Download

Save as `rita_input/orders-DDmmm.csv` (e.g. `orders-02apr.csv`).

Currently used for audit trail only — not yet wired into any dashboard calc.

---

## 6. Review scenario levels

Open `rita_input/scenario_levels.csv`:

```
underlying,mode,sl,target,ledger_balance
NIFTY,bull,22700,24300,3500000
NIFTY,bear,23531,21500,
BANKNIFTY,bull,52000,56500,
BANKNIFTY,bear,54600,49100,
```

Check:
- Is today's NIFTY close **above** the bull SL (`22700`)? If not, bull scenario is invalidated — update or remove.
- Is today's BANKNIFTY close **above** the bull SL (`52000`)? Same check.
- Update `ledger_balance` on the NIFTY bull row if your account balance changed.

> Scenario levels are gitignored — safe to edit directly.

---

## 7. Record any closed trades

If you closed or let expire any options/futures today, append rows to `rita_input/closed_positions.csv`:

```
instrument,expiry,type,strike,underlying,side,pnl,notes
BANKNIFTY 50000 PE,26-Apr-26,PE,50000,BANKNIFTY,Long,-4500,Closed 02-Apr
```

---

## 8. Verify in the dashboard

```powershell
cd C:\Users\Sandeep\Documents\Work\code\poc\rita-cowork-demo
. .\activate-env.ps1
python run_api.py
```

Open `http://localhost:8000/dashboard/` and check:
- RITA main page: NIFTY ATR / Trend Score reflects today's close
- FnO page: positions, P&L, scenario levels look correct
- Manoeuvre page: today's date appears, unrealized P&L updated

---

## Quick reference — file formats

| File | Date format | Numbers |
|------|-------------|---------|
| `rita_output/nifty_merged.csv` | `YYYY-MM-DD 00:00:00+05:30` | Plain decimals |
| `rita_input/nifty_manual.csv` | `DD-MMM-YYYY` (e.g. `02-APR-2026`) | Plain decimals |
| `rita_input/banknifty_manual.csv` | `DD-MMM-YYYY` | Plain decimals |
| `rita_input/positions-DDmmm.csv` | NSE export as-is | As exported |

**Indian comma formatting warning:** NSE sometimes exports numbers like `22,824.35`. Always fix to `22824.35` before saving to manual CSVs. The merged CSV and `prepare_data()` do not handle this format.
