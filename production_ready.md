# RITA POC → Production Readiness Assessment

> **Document Version:** 2.0 — Updated 29-Mar-2026 with architectural direction
> **Scope:** Analysis of what is required to move RITA from a working POC to a production-grade financial portal, including two-version migration strategy, API tier design, and cloud deployment architecture.

---

## Executive Summary

RITA is a Python FastAPI + HTML/JS financial trading dashboard with a Double DQN RL model. As a POC it validates the domain logic and user workflows. Moving to production requires addressing critical gaps in security, data integrity, API design, and deployment — but the migration will follow a deliberate two-version strategy rather than a single big-bang rewrite.

### Architectural Pillars for Production

Three non-negotiable decisions govern all design choices:

**1. CSV-first, DB-ready (v1 → v2 migration path)**
Version 1 retains CSV persistence but every CSV is designed as a direct projection of a normalised relational table. The column names, data types, primary keys, and foreign-key relationships are identical to what will become SQL schema in v2. Migration from CSV to PostgreSQL becomes a mechanical `COPY` command per table, not a data restructure.

**2. Three-tier API architecture (implemented from v1)**
The single `rest_api.py` monolith is replaced by three distinct API layers deployed and versioned independently:
- **Experience APIs** — thin, client-facing aggregation; one per UI surface
- **Business Process APIs** — workflow orchestration, calculations, rules
- **System APIs** — CRUD operations on domain objects (positions, orders, sessions)

**3. Cloud-native deployment**
The application is designed for a managed cloud platform from day one: containerised workloads behind a load balancer, stateless API pods, external shared storage, secrets in a vault, and a full CI/CD pipeline with automated tests, build, and progressive rollout.

### Key Risk Areas (existing POC)

| # | Risk | Severity |
|---|------|----------|
| 1 | No real authentication — API key hardcoded in HTML | CRITICAL |
| 2 | CSV writes with race conditions under concurrent access | CRITICAL |
| 3 | All business logic, routing, and DB access in one 1,533-line file | HIGH |
| 4 | CORS open to `*` — any origin can call the API | HIGH |
| 5 | Silent NaN→null conversion hides calculation errors | HIGH |
| 6 | Minimal test coverage (~1%) — no integration tests | HIGH |
| 7 | Hardcoded `localhost:8000` in dashboard HTML | HIGH |
| 8 | No structured logging — can't diagnose production issues | MEDIUM |
| 9 | No input validation on user-supplied dates and parameters | MEDIUM |
| 10 | Model versioning by manual filename rename | MEDIUM |

### Versioning Roadmap

| Version | Theme | Key Changes |
|---------|-------|-------------|
| **v1** | Production-hardened POC | Three-tier API, CSV normalised as tables, auth, logging, CI/CD, cloud deploy |
| **v2** | Database migration | PostgreSQL replaces CSVs (mechanical migration, no logic change) |
| **v3** | Scale & intelligence | ML-driven suggestions, real-time streaming, multi-user RBAC |

**Estimated effort:**
- v1: 10–12 weeks (security + API restructure + cloud infra + test coverage)
- v2: 3–4 weeks (DB migration + ORM integration)

---

## 1. API Tier Architecture

### Current State

All 24+ endpoints live in `src/rita/interfaces/rest_api.py` (1,533 lines). Routes mix concerns: a single endpoint may read a CSV, run a calculation, aggregate data for the UI, and write an audit record. There is no separation between what the client needs to see, what business logic is involved, and what domain data is being operated on.

### Target: Three-Tier API Design

```
┌─────────────────────────────────────────────────────┐
│                    Clients                          │
│  HTML Dashboard  │  Streamlit  │  MCP / Claude      │
└─────────┬────────┴──────┬──────┴──────┬─────────────┘
          │               │             │
          ▼               ▼             ▼
┌─────────────────────────────────────────────────────┐
│              EXPERIENCE APIs  (BFF Layer)            │
│  /bff/dashboard/*  /bff/fno/*  /bff/ops/*           │
│                                                     │
│  • Aggregates data from multiple System APIs        │
│  • Shapes response to exactly what each UI needs    │
│  • No business logic — only composition & formatting│
│  • One BFF per distinct UI surface                  │
└─────────────────────┬───────────────────────────────┘
                      │  internal calls only
          ┌───────────┴─────────────┐
          ▼                         ▼
┌─────────────────┐     ┌───────────────────────────┐
│ BUSINESS PROCESS│     │       SYSTEM APIs          │
│      APIs       │     │                            │
│ /api/v1/workflow│     │ /api/v1/positions          │
│ /api/v1/backtest│     │ /api/v1/orders             │
│ /api/v1/train   │     │ /api/v1/sessions           │
│ /api/v1/analyse │     │ /api/v1/snapshots          │
│                 │     │ /api/v1/scenario-levels    │
│ • Orchestration │     │ /api/v1/model-registry     │
│ • Calculations  │     │                            │
│ • Rules & gates │     │ • Pure CRUD on domain objs │
│ • Async tasks   │     │ • No business logic        │
│ • Event pub/sub │     │ • Schema = normalised table│
└─────────────────┘     └───────────────────────────┘
          │                         │
          └───────────┬─────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              DATA LAYER (CSV v1 / DB v2)            │
│  CSV files named as tables, columns as DB columns   │
└─────────────────────────────────────────────────────┘
```

### Experience APIs (BFF)

Each UI surface gets one BFF module. The BFF knows what that page needs and fetches it from System APIs. It never touches CSVs or runs calculations directly.

```
src/rita/interfaces/bff/
  dashboard_bff.py     # GET /bff/dashboard/summary — aggregates RL model state + recent backtest
  fno_bff.py           # GET /bff/fno/portfolio     — positions + greeks + scenario payoffs + manoeuvre
  ops_bff.py           # GET /bff/ops/status        — snapshot status + action counts + notes
```

**Example — FnO BFF endpoint:**
```python
# src/rita/interfaces/bff/fno_bff.py

@router.get("/bff/fno/portfolio")
async def fno_portfolio(und: str = "NIFTY", month: str = "APR"):
    """
    Single call for the full FnO dashboard load.
    Aggregates: positions, greeks, scenario payoffs, manoeuvre group state, market data.
    """
    positions   = await system_api.positions.list(und=und, month=month, status="open")
    greeks      = await bp_api.analytics.compute_greeks(positions)
    scenarios   = await system_api.scenario_levels.get(und=und)
    groups      = await system_api.manoeuvre_groups.get(und=und, month=month)
    market      = await system_api.market_data.latest(und=und)

    return {
        "positions":       positions,
        "greeks":          greeks,
        "scenario_levels": scenarios,
        "manoeuvre":       groups,
        "market":          market,
    }
```

**Rules for BFF layer:**
- Never import `portfolio_manager.py` or `risk_engine.py` directly
- Always calls System or Business Process APIs via internal HTTP or shared service interface
- Response shape is driven by client contract (what the page needs), not DB schema
- Versioned separately: `/bff/v1/fno/portfolio`, `/bff/v2/fno/portfolio`

### Business Process APIs

These own orchestration, calculations, and multi-step workflows. They call System APIs to read/write data, then apply business logic.

```
src/rita/interfaces/bp/
  workflow_bp.py       # POST /api/v1/workflow/run      — 8-step pipeline
  backtest_bp.py       # POST /api/v1/backtest/run      — rolling backtest
  training_bp.py       # POST /api/v1/model/train       — RL training run
  analytics_bp.py      # POST /api/v1/analytics/greeks  — Greek computation
  manoeuvre_bp.py      # POST /api/v1/manoeuvre/snapshot — snapshot + CSV writes
```

**Example — Manoeuvre snapshot Business Process:**
```python
# src/rita/interfaces/bp/manoeuvre_bp.py

@router.post("/api/v1/manoeuvre/snapshot")
async def save_snapshot(payload: ManSnapshotRequest):
    """
    Business process: validate → compute payoffs → write history + position snapshot + notes.
    All writes are transactional (v1: file-level lock; v2: DB transaction).
    """
    # 1. Validate
    positions = await positions_svc.list(und=payload.und, month=payload.month, status="open")
    if not positions:
        raise HTTPException(422, f"No open {payload.und} {payload.month} positions to snapshot")

    # 2. Compute payoffs (Business Logic)
    scenarios = await scenario_svc.get(und=payload.und)
    enriched_groups = payoff_calculator.enrich(payload.groups, scenarios)

    # 3. Write (System API calls — v1 writes CSV, v2 writes DB rows)
    await snapshot_svc.upsert(
        date=payload.date, und=payload.und, month=payload.month,
        groups=enriched_groups
    )
    await position_snapshot_svc.upsert(...)
    if payload.notes:
        await session_notes_svc.append(...)

    return {"status": "ok", "rows_written": len(enriched_groups)}
```

### System APIs

Pure domain object CRUD. No business logic. Column names mirror DB columns. Schema changes here propagate identically to the CSV structure and later to the DB migration.

```
src/rita/interfaces/system/
  positions_api.py          # GET/POST/PATCH /api/v1/positions
  orders_api.py             # GET/POST       /api/v1/orders
  sessions_api.py           # GET/POST/PATCH /api/v1/sessions
  snapshots_api.py          # GET/POST       /api/v1/snapshots/pnl-history
  scenario_levels_api.py    # GET/PUT        /api/v1/scenario-levels
  manoeuvre_groups_api.py   # GET/PUT        /api/v1/manoeuvre-groups
  model_registry_api.py     # GET/POST       /api/v1/model-registry
```

**Example — Snapshot System API (identical interface in v1 CSV and v2 DB):**
```python
# src/rita/interfaces/system/snapshots_api.py

class SnapshotService:
    """
    Interface is stable. Implementation swaps between CSV (v1) and DB (v2).
    """
    async def upsert(self, date: str, und: str, month: str,
                     groups: list[GroupSnapshot]) -> int:
        raise NotImplementedError

class CsvSnapshotService(SnapshotService):
    """v1 — writes man_pnl_history.csv"""
    async def upsert(self, date, und, month, groups):
        # read CSV, filter out (date, und, month) rows, append new rows, write
        ...

class DbSnapshotService(SnapshotService):
    """v2 — writes man_pnl_history table"""
    async def upsert(self, date, und, month, groups):
        async with db.transaction():
            await db.execute("""
                DELETE FROM man_pnl_history
                WHERE date=$1 AND und=$2 AND month=$3
            """, date, und, month)
            await db.executemany("""
                INSERT INTO man_pnl_history (date, und, month, group_id, ...)
                VALUES ($1, $2, $3, $4, ...)
            """, [(date, und, month, g.id, ...) for g in groups])
```

**Dependency injection selects implementation at startup:**
```python
# src/rita/config.py
def make_snapshot_service() -> SnapshotService:
    if settings.USE_DATABASE:
        return DbSnapshotService(db=settings.DB_URL)
    return CsvSnapshotService(output_dir=settings.OUTPUT_DIR)
```

This is the mechanism that makes v1→v2 a configuration change, not a code rewrite.

### Backend File Structure (Target)

The current repo has entry points (`run_api.py`, `run_ui.py`, `run_pipeline.py`) scattered at root, core logic mixed with interface code inside `src/rita/`, and no clear separation between calculators, services, and data access. The target structure enforces clean layering that mirrors the three-tier API design.

```
rita-cowork/                              ← repo root
│
├── pyproject.toml                        # single source of truth for deps + build
├── .env.example                          # template — never commit .env
├── run_api.py                            # thin launcher — uvicorn only, no logic
├── run_ui.py                             # thin launcher — streamlit only
│
├── config/                               # NEW: environment configs
│   ├── base.yaml                         # shared defaults
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
│
├── k8s/                                  # NEW: Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── volumes.yaml
│   └── alerting-rules.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yml                        # quality gate on every push
│       └── deploy.yml                    # build + staging + prod on main merge
│
├── scripts/                              # NEW: one-off operational scripts
│   ├── migrate_csv_to_db.py             # v2 migration
│   ├── backfill_snapshot_ids.py
│   └── validate_csv_schemas.py
│
├── tests/
│   ├── unit/
│   │   ├── test_greeks.py               # Black-Scholes reference values
│   │   ├── test_payoff_calculator.py
│   │   ├── test_csv_repositories.py
│   │   ├── test_scenario_service.py
│   │   └── test_config.py
│   ├── integration/
│   │   ├── test_manoeuvre_snapshot.py   # API → CSV → read-back
│   │   ├── test_workflow_pipeline.py
│   │   └── conftest.py                  # tmp_path fixtures, test client
│   ├── e2e/                             # Playwright, runs against staging
│   │   ├── test_fno_portfolio.py
│   │   └── test_daily_ops.py
│   └── fixtures/
│       ├── positions_sample.csv         # realistic test data
│       ├── orders_sample.csv
│       └── scenario_levels_sample.csv
│
└── src/rita/
    │
    ├── config.py                         # Pydantic Settings — validated at import
    ├── logger.py                         # structlog setup
    ├── exceptions.py                     # domain exception hierarchy
    │
    ├── core/                             # UNCHANGED: pure calculation, no I/O
    │   ├── portfolio_manager.py          # Greeks, margin, payoff
    │   ├── risk_engine.py               # VaR, CVaR, drawdown
    │   ├── technical_analyzer.py        # RSI, MACD, BB, ATR
    │   ├── rl_agent.py                  # DDQN env + stable-baselines3
    │   ├── performance.py               # Sharpe, MDD, CAGR
    │   ├── data_loader.py
    │   ├── goal_engine.py
    │   ├── strategy_engine.py
    │   ├── classifier.py
    │   └── chat_monitor.py
    │
    ├── services/                         # NEW: business rules, no HTTP, no I/O
    │   ├── payoff_calculator.py          # at-expiry + scenario payoffs
    │   ├── greeks_service.py             # wraps core/portfolio_manager Greeks
    │   ├── scenario_service.py           # scenario level loading + validation
    │   ├── model_registry.py             # active model selection + versioning
    │   ├── lot_service.py               # lot expansion, lot size config
    │   └── audit_service.py             # writes to audit_log
    │
    ├── repositories/                     # NEW: all data access behind interfaces
    │   ├── base.py                       # abstract repo interfaces
    │   ├── csv/                          # v1 implementations
    │   │   ├── positions_repo.py
    │   │   ├── orders_repo.py
    │   │   ├── snapshots_repo.py
    │   │   ├── manoeuvre_groups_repo.py
    │   │   ├── session_repo.py
    │   │   ├── model_registry_repo.py
    │   │   └── audit_repo.py
    │   └── db/                           # v2 implementations (stub in v1)
    │       ├── positions_repo.py
    │       ├── orders_repo.py
    │       └── ...
    │
    ├── interfaces/
    │   ├── app.py                        # FastAPI factory — mounts all routers
    │   │
    │   ├── bff/                          # Experience APIs — one per UI surface
    │   │   ├── dashboard_bff.py          # /bff/dashboard/* — RL + backtest
    │   │   ├── fno_bff.py               # /bff/fno/*       — portfolio + Greeks
    │   │   └── ops_bff.py               # /bff/ops/*       — snapshot status
    │   │
    │   ├── bp/                           # Business Process APIs
    │   │   ├── workflow_bp.py            # /api/v1/workflow/*
    │   │   ├── backtest_bp.py            # /api/v1/backtest/*
    │   │   ├── training_bp.py            # /api/v1/model/train
    │   │   ├── analytics_bp.py           # /api/v1/analytics/greeks
    │   │   └── manoeuvre_bp.py           # /api/v1/manoeuvre/*
    │   │
    │   ├── system/                       # System APIs — domain object CRUD
    │   │   ├── positions_api.py          # /api/v1/positions
    │   │   ├── orders_api.py             # /api/v1/orders
    │   │   ├── sessions_api.py           # /api/v1/sessions
    │   │   ├── snapshots_api.py          # /api/v1/snapshots
    │   │   ├── scenario_levels_api.py    # /api/v1/scenario-levels
    │   │   ├── manoeuvre_groups_api.py   # /api/v1/manoeuvre-groups
    │   │   ├── model_registry_api.py     # /api/v1/model-registry
    │   │   └── health_api.py             # /health, /health/ready
    │   │
    │   ├── streamlit_app.py              # Streamlit UI (internal/analyst tool)
    │   └── mcp_server.py                 # Claude MCP tool server
    │
    └── orchestration/
        ├── workflow.py                   # 8-step pipeline runner
        ├── session.py                    # workflow session state
        └── monitor.py                    # phase timing + CSV logging
```

**Key rules enforced by this structure:**
- `core/` never imports from `services/`, `repositories/`, or `interfaces/`
- `services/` imports from `core/` and `repositories/` only — no HTTP, no FastAPI
- `repositories/` imports from nothing in `rita/` — only stdlib + pandas
- `interfaces/bff/` imports from `services/` — never from `core/` or `repositories/` directly
- `interfaces/system/` imports from `repositories/` only — no business logic

---

## 2. Data Layer — CSV as Normalised Tables (v1) → Database (v2)

### Design Principle

Every CSV file is a projection of a normalised database table. Column names are identical to what will become SQL column names. Primary keys and foreign keys are present as plain columns. No "formatted" values, no merged cells, no multi-value columns. A `pd.read_csv()` call followed by `df.to_sql()` is sufficient to migrate each file.

### Normalised Table Design

#### Core Domain Tables

**`positions` (rita_input/positions.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `position_id` | TEXT PK | Broker-assigned unique ID |
| `instrument` | TEXT | NSE symbol e.g. NIFTY25APR22700CE |
| `und` | TEXT | NIFTY \| BANKNIFTY |
| `exp` | TEXT | APR \| MAY \| JUN |
| `type` | TEXT | CE \| PE \| FUT |
| `strike_val` | NUMERIC | Strike price (0 for FUT) |
| `side` | TEXT | Long \| Short |
| `qty` | INTEGER | Total quantity (lots × lot_size) |
| `avg` | NUMERIC | Average entry price |
| `ltp` | NUMERIC | Last traded price |
| `pnl` | NUMERIC | Unrealised P&L in INR |
| `loaded_at` | TIMESTAMP | When this file was ingested |

**`orders` (rita_input/orders.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `order_id` | TEXT PK | Broker order ID |
| `position_id` | TEXT FK→positions | Links to open position |
| `instrument` | TEXT | NSE symbol |
| `order_type` | TEXT | BUY \| SELL |
| `qty` | INTEGER | |
| `price` | NUMERIC | Fill price |
| `order_at` | TIMESTAMP | Execution timestamp |
| `loaded_at` | TIMESTAMP | When ingested |

**`scenario_levels` (rita_input/scenario_levels.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `und` | TEXT PK part | NIFTY \| BANKNIFTY |
| `mode` | TEXT PK part | bull \| bear |
| `sl` | NUMERIC | Stop-loss index level |
| `target` | NUMERIC | Target index level |
| `ledger_balance` | NUMERIC | Usable margin (INR) — nullable |
| `updated_at` | TIMESTAMP | When last changed |

#### Manoeuvre Tables

**`manoeuvre_group_assignments` (rita_output/man_groups_{UND}_{MONTH}.csv)**

Currently stored as JSON. Normalise to CSV:

| Column | Type | Notes |
|--------|------|-------|
| `und` | TEXT PK part | |
| `month` | TEXT PK part | |
| `group_id` | TEXT PK part | anchor \| directional \| futures \| spread \| hedge |
| `group_name` | TEXT | User-overridden display name |
| `view` | TEXT | bull \| bear |
| `saved_at` | TIMESTAMP | |

**`manoeuvre_lot_assignments` (rita_output/man_lot_assignments.csv)**

Currently embedded in JSON. Normalise:

| Column | Type | Notes |
|--------|------|-------|
| `und` | TEXT PK part | |
| `month` | TEXT PK part | |
| `lot_key` | TEXT PK part | e.g. NIFTY25APR22700CE_L1 |
| `group_id` | TEXT FK | |
| `assigned_at` | TIMESTAMP | |

**`man_pnl_history` (rita_output/man_pnl_history.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `snapshot_id` | TEXT PK | `{date}_{und}_{month}_{group_id}` |
| `date` | DATE | |
| `und` | TEXT | |
| `month` | TEXT | |
| `group_id` | TEXT | |
| `group_name` | TEXT | |
| `view` | TEXT | |
| `pnl_now` | NUMERIC | INR |
| `sl_pnl` | NUMERIC | INR at SL level |
| `target_pnl` | NUMERIC | INR at target level |
| `lot_count` | INTEGER | |
| `nifty_spot` | NUMERIC | |
| `banknifty_spot` | NUMERIC | |
| `dte` | INTEGER | |
| `pct_from_sl` | NUMERIC | Signed % |
| `pct_from_target` | NUMERIC | Signed % |
| `created_at` | TIMESTAMP | When snapshot was saved |

**`man_position_snapshot` (rita_output/man_position_snapshot.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `snapshot_id` | TEXT FK→man_pnl_history | |
| `lot_key` | TEXT PK part | |
| `date` | DATE | |
| `und` | TEXT | |
| `month` | TEXT | |
| `group_id` | TEXT | |
| `instrument` | TEXT | |
| `type` | TEXT | |
| `side` | TEXT | |
| `lot_sz` | INTEGER | |
| `avg` | NUMERIC | |
| `pnl_now` | NUMERIC | |
| `pnl_sl` | NUMERIC | |
| `pnl_target` | NUMERIC | |

**`man_action_log` (rita_output/man_action_log.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `action_id` | TEXT PK | UUID |
| `ts` | TIMESTAMP PK part | |
| `date` | DATE | |
| `und` | TEXT | |
| `month` | TEXT | |
| `action` | TEXT | assign \| unassign \| remove |
| `lot_key` | TEXT | |
| `from_group` | TEXT | Empty = pool |
| `to_group` | TEXT | Empty = pool/removed |
| `nifty_spot` | NUMERIC | |
| `banknifty_spot` | NUMERIC | |

**`man_session_notes` (rita_output/man_session_notes.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `note_id` | TEXT PK | UUID |
| `ts` | TIMESTAMP | |
| `date` | DATE | |
| `und` | TEXT | |
| `month` | TEXT | |
| `nifty_spot` | NUMERIC | |
| `banknifty_spot` | NUMERIC | |
| `dte` | INTEGER | |
| `notes` | TEXT | Free text |

#### RL / Workflow Tables

**`workflow_sessions` (rita_output/session_state.csv → normalised)**

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | TEXT PK | UUID |
| `status` | TEXT | pending \| running \| completed \| failed |
| `created_at` | TIMESTAMP | |
| `completed_at` | TIMESTAMP | nullable |
| `goal_json` | TEXT | Serialised JSON blob |
| `research_json` | TEXT | nullable |
| `strategy_json` | TEXT | nullable |

**`backtest_daily` (rita_output/backtest_daily.csv)**

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | TEXT FK→workflow_sessions | |
| `date` | DATE | |
| `close_price` | NUMERIC | |
| `portfolio_value` | NUMERIC | INR |
| `benchmark_value` | NUMERIC | INR |
| `allocation_pct` | INTEGER | 0 \| 50 \| 100 |
| `signal` | TEXT | BUY \| SELL \| HOLD |
| `realized_pnl` | NUMERIC | |

**`model_registry` (rita_output/model_registry.csv — new)**

| Column | Type | Notes |
|--------|------|-------|
| `model_id` | TEXT PK | UUID |
| `trained_at` | TIMESTAMP | |
| `zip_path` | TEXT | Relative path to .zip file |
| `val_sharpe` | NUMERIC | |
| `val_mdd_pct` | NUMERIC | |
| `timesteps` | INTEGER | |
| `seed` | INTEGER | |
| `train_start` | DATE | |
| `train_end` | DATE | |
| `is_active` | BOOLEAN | Only one row can be TRUE |

### CSV Integrity Rules (v1)

To catch data quality issues before they reach calculations, add these rules in the repository layer:

```python
# src/rita/repositories/csv/base_csv_repo.py

class CsvRepository:
    def _validate_schema(self, df: pd.DataFrame, required_cols: list[str]):
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise SchemaError(f"Missing columns: {missing}")

    def _check_pk_uniqueness(self, df: pd.DataFrame, pk_cols: list[str]):
        dupes = df[df.duplicated(pk_cols, keep=False)]
        if not dupes.empty:
            raise DataIntegrityError(f"Duplicate primary keys found: {dupes[pk_cols].values.tolist()[:5]}")

    def _check_no_nulls(self, df: pd.DataFrame, not_null_cols: list[str]):
        for col in not_null_cols:
            if df[col].isna().any():
                raise DataIntegrityError(f"NULL values in non-nullable column: {col}")
```

### v2 Migration (when ready)

When the business decides to move to DB, the migration is:

```python
# scripts/migrate_csv_to_db.py
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(DB_URL)
TABLE_MAP = {
    "rita_output/man_pnl_history.csv":       "man_pnl_history",
    "rita_output/man_position_snapshot.csv": "man_position_snapshot",
    "rita_output/man_action_log.csv":        "man_action_log",
    "rita_output/man_session_notes.csv":     "man_session_notes",
    "rita_output/backtest_daily.csv":        "backtest_daily",
    "rita_output/model_registry.csv":        "model_registry",
}

for csv_path, table in TABLE_MAP.items():
    df = pd.read_csv(csv_path, parse_dates=True)
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"Migrated {len(df)} rows → {table}")
```

Then set `USE_DATABASE=true` in config. The Service layer sees `DbSnapshotService` instead of `CsvSnapshotService`. No business logic changes.

---

## 3. Cloud Architecture & Deployment

### Target Architecture

```
Internet
    │
    ▼
┌───────────────────────────────────────────────────────┐
│            Cloud Load Balancer (HTTPS)                │
│   SSL termination · WAF · DDoS protection             │
│   Health check: GET /health                           │
└─────────────────────┬─────────────────────────────────┘
                      │
          ┌───────────┴──────────┐
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  BFF / API Pod  │    │  BFF / API Pod  │  ← Horizontal scaling
│  (FastAPI)      │    │  (FastAPI)      │
│  Port 8000      │    │  Port 8000      │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
        ┌───────────┴──────────────────┐
        │                              │
        ▼                              ▼
┌───────────────┐            ┌─────────────────────┐
│  File Storage │            │  Secrets Manager     │
│  (S3 / GCS)   │            │  (Vault / AWS SM)    │
│  rita_output/ │            │  API keys, DB URL    │
│  rita_input/  │            │  Broker credentials  │
└───────────────┘            └─────────────────────┘
        │
        ▼
┌───────────────┐  ← v2 only
│  PostgreSQL   │
│  (RDS/Cloud   │
│   SQL)        │
└───────────────┘

Observability (all environments):
┌─────────────────────────────────────────────────────┐
│  Structured logs → CloudWatch / GCP Logging         │
│  Metrics → Prometheus + Grafana                     │
│  Traces → OpenTelemetry (Jaeger / Cloud Trace)      │
│  Alerts → AlertManager / PagerDuty                  │
│  Error tracking → Sentry                            │
└─────────────────────────────────────────────────────┘
```

### Stateless API Design

Every API pod must be fully stateless — no local file writes, no in-memory state shared across requests. This enables horizontal scaling and zero-downtime rollouts.

**What this means for RITA:**

| Current (POC) | v1 Production |
|---------------|---------------|
| Writes CSVs to local `rita_output/` | Writes CSVs to mounted shared volume (S3-backed EFS / GCS Filestore) |
| `rita_input/` is local filesystem | Shared volume, writable only by ingestion pod |
| Model `.zip` is local file | Read from shared volume; loaded once at startup |
| Session state in memory + CSV | CSV on shared volume (v1); DB (v2) |
| Orchestrator singleton warms up at start | Per-request orchestrator with lightweight warm path |

**Shared volume mount (Kubernetes):**
```yaml
# k8s/volumes.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rita-data-pvc
spec:
  storageClassName: efs-sc     # or gce-pd / azure-file
  accessModes:
    - ReadWriteMany             # All pods share the same volume
  resources:
    requests:
      storage: 50Gi
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rita-api
  labels:
    app: rita-api
    version: v1
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0         # Zero-downtime rollout
  selector:
    matchLabels:
      app: rita-api
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/org/rita:${IMAGE_TAG}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: OUTPUT_DIR
          value: /data/rita_output
        - name: INPUT_DIR
          value: /data/rita_input
        - name: PORTFOLIO_API_KEY
          valueFrom:
            secretKeyRef:
              name: rita-secrets
              key: portfolio_api_key
        - name: DATABASE_URL        # v2
          valueFrom:
            secretKeyRef:
              name: rita-secrets
              key: database_url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 15
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: rita-data-pvc
```

### Health Endpoints

```python
# src/rita/interfaces/system/health_api.py

@router.get("/health")
async def liveness():
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive", "version": settings.VERSION, "ts": utcnow()}

@router.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe — can this pod serve traffic?"""
    checks = {}

    # Data volume accessible
    try:
        Path(settings.OUTPUT_DIR).stat()
        checks["data_volume"] = "ok"
    except OSError as e:
        checks["data_volume"] = f"error: {e}"

    # Model loaded
    checks["model"] = "ok" if model_registry.is_loaded() else "not_loaded"

    # DB ping (v2 only)
    if settings.USE_DATABASE:
        try:
            await db.execute("SELECT 1")
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse({"status": "ready" if all_ok else "degraded", "checks": checks},
                        status_code=status_code)
```

### Resiliency Patterns

**Circuit breaker for external calls:**
```python
# src/rita/services/circuit_breaker.py
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def fetch_market_data(und: str) -> dict:
    """Retry with exponential backoff on transient failures."""
    response = await http_client.get(f"{MARKET_DATA_URL}/{und}")
    response.raise_for_status()
    return response.json()
```

**Timeout on all external I/O:**
```python
async with asyncio.timeout(5.0):     # 5-second hard timeout
    data = await fetch_market_data("NIFTY")
```

**Graceful shutdown:**
```python
# src/rita/interfaces/app.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await model_registry.load()
    logger.info("RITA API ready", extra={"version": settings.VERSION})
    yield
    # Shutdown — drain in-flight requests, close connections
    logger.info("Shutting down — draining requests")
    await asyncio.sleep(2)           # let load balancer drain
    await db.close()                 # v2
    logger.info("Shutdown complete")
```

---

## 4. CI/CD Pipeline

### Pipeline Overview

```
Developer pushes branch
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Stage 1: QUALITY GATE (every push)                   │
│  ruff lint → mypy type check → pytest (unit tests)    │
│  coverage ≥ 80% gate → security scan (bandit)         │
│  Blocks merge if any fail                             │
└─────────────────────────┬─────────────────────────────┘
                          │ on PR merge to main
                          ▼
┌───────────────────────────────────────────────────────┐
│  Stage 2: BUILD & INTEGRATION                         │
│  Docker multi-stage build → pytest integration tests  │
│  Push image to registry (tagged with git SHA)         │
└─────────────────────────┬─────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────┐
│  Stage 3: DEPLOY TO STAGING                           │
│  kubectl rollout (RollingUpdate, maxUnavailable=0)    │
│  Smoke tests against staging URL                      │
│  Manual approval gate for production                  │
└─────────────────────────┬─────────────────────────────┘
                          │ on approval
                          ▼
┌───────────────────────────────────────────────────────┐
│  Stage 4: DEPLOY TO PRODUCTION                        │
│  Blue/green or canary deploy (10% → 50% → 100%)       │
│  Automatic rollback if error rate > 1% in 5 min       │
│  Notify Slack: deploy complete + metrics link         │
└───────────────────────────────────────────────────────┘
```

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/rita-api

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: mypy src/rita --ignore-missing-imports
      - run: bandit -r src/rita -ll           # security scan
      - run: |
          pytest tests/ \
            --cov=src/rita \
            --cov-report=xml \
            --cov-fail-under=80 \
            -v
      - uses: codecov/codecov-action@v4

  build:
    needs: quality
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    environment: staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        run: |
          kubectl set image deployment/rita-api \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --namespace=staging
          kubectl rollout status deployment/rita-api --namespace=staging --timeout=120s
      - name: Smoke test
        run: |
          curl -f https://rita-staging.example.com/health/ready || exit 1

  deploy-production:
    needs: deploy-staging
    environment: production           # requires manual approval in GitHub
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production (canary)
        run: |
          kubectl set image deployment/rita-api \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --namespace=production
          kubectl rollout status deployment/rita-api --namespace=production --timeout=300s
```

### Dockerfile (Production-grade)

```dockerfile
# Build + test stage
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .
RUN pytest tests/ --cov=src/rita --cov-fail-under=80 -q

# Runtime stage (no test dependencies, no root)
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends dumb-init curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 rita

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

COPY --from=builder /app/src ./src
COPY run_api.py .

RUN mkdir -p /data/rita_output /data/rita_input \
    && chown -R rita:rita /data

USER rita

ENV OUTPUT_DIR=/data/rita_output \
    INPUT_DIR=/data/rita_input \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "run_api.py"]
```

---

## 5. Security

### Current State

| Vulnerability | Location | Severity |
|--------------|----------|----------|
| API key hardcoded in dashboard HTML | `dashboard/fno.html` line ~45: `const API_KEY = '...'` | CRITICAL |
| CORS open to all origins `*` | `rest_api.py` line 78 | HIGH |
| Auth is optional — `_require_portfolio_key` only on some routes | `rest_api.py` | HIGH |
| No rate limiting | All endpoints | HIGH |
| No request size limit | All endpoints | MEDIUM |
| Input values not validated (dates, months, quantities) | Manoeuvre endpoints | MEDIUM |
| Secrets in environment variables without rotation | `config.py` | MEDIUM |

### Recommendations

**1. Replace API key with JWT + secure cookie:**
```python
# src/rita/services/auth_service.py
from fastapi import Depends, HTTPException, Cookie
from jose import JWTError, jwt

def get_current_user(token: str = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Applied to all protected routes
@router.get("/bff/fno/portfolio", dependencies=[Depends(get_current_user)])
async def fno_portfolio(...):
    ...
```

**2. Restrict CORS to known origins:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,    # ["https://rita.example.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
)
```

**3. Rate limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/api/v1/manoeuvre/snapshot")
@limiter.limit("10/minute")
async def save_snapshot(...):
    ...
```

**4. Input validation with Pydantic models (not raw `dict`):**
```python
class ManSnapshotRequest(BaseModel):
    month: str = Field(..., pattern=r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$")
    und: str = Field(..., pattern=r"^(NIFTY|BANKNIFTY)$")
    date: str = Field(..., pattern=r"^\d{2}-[A-Z][a-z]{2}-\d{4}$")
    nifty_spot: float = Field(..., gt=10000, lt=50000)
    banknifty_spot: float = Field(..., gt=20000, lt=100000)
    groups: list[GroupSnapshot] = Field(..., min_length=1, max_length=10)
```

**5. Secrets in cloud vault (never in env vars for prod):**
```python
# AWS Secrets Manager / GCP Secret Manager
import boto3

def load_secrets():
    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId="rita/production")
    return json.loads(secret["SecretString"])
```

---

## 6. Error Handling & Resilience

### Current State — Silent Failures

Many exception handlers swallow errors without logging:
```python
# dashboard/fno-manoeuvre.js (fire-and-forget action log)
function manLogAction(...) {
    fetch('/api/v1/portfolio/man-action', {...}).catch(() => {});  # swallowed silently
}

# rest_api.py — NaN silently becomes null
def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None  # ← no log, no alert
```

### Structured Error Responses

All API errors must return consistent shape with a trace ID:
```python
# src/rita/interfaces/app.py
import uuid

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error("Unhandled exception", extra={
        "trace_id": trace_id,
        "path": request.url.path,
        "error": str(exc),
    }, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "trace_id": trace_id}
    )
```

### NaN/Inf Handling — Log, Don't Swallow

```python
def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        logger.warning("NaN/Inf detected in financial calculation",
                       extra={"value": obj, "type": type(obj).__name__})
        return None   # still null in JSON, but now logged
    return obj
```

---

## 7. Observability & Monitoring

### Structured Logging

Replace all `print()` statements with structured JSON logs:

```python
# src/rita/logger.py
import structlog

logger = structlog.get_logger()

# Usage in any module:
logger.info("snapshot_saved",
            und=und, month=month, group_count=len(groups),
            rows_written=rows_written, trace_id=trace_id)

logger.error("snapshot_failed",
             und=und, month=month, error=str(e),
             exc_info=True, trace_id=trace_id)
```

### Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, make_asgi_app

snapshot_counter = Counter("rita_snapshots_total", "Total snapshots saved",
                            ["und", "month", "status"])
request_latency  = Histogram("rita_request_duration_seconds", "Request latency",
                             ["method", "endpoint"])

# In the snapshot endpoint:
snapshot_counter.labels(und=und, month=month, status="success").inc()
```

### Alerts

```yaml
# k8s/alerting-rules.yaml
groups:
- name: rita.rules
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
    for: 2m
    annotations:
      summary: "RITA API error rate > 1%"

  - alert: SnapshotMissed
    expr: rita_snapshots_total{status="success"} == 0
    for: 1h
    annotations:
      summary: "No snapshots saved in last hour — check scheduled trigger"

  - alert: ApiPodDown
    expr: kube_deployment_status_replicas_available{deployment="rita-api"} < 1
    for: 1m
    annotations:
      summary: "All RITA API pods are down"
```

---

## 8. Testing Strategy

### Current State

`tests/test_core.py` — 376 lines, synthetic fixtures, no real CSV fixtures, no API tests, no integration tests. Coverage is below 5%.

### Target Test Pyramid

```
              ┌─────────┐
              │   E2E   │  5 tests — full user journeys via browser
              │  Tests  │  (Playwright, staging env)
              └────┬────┘
          ┌────────┴────────┐
          │  Integration    │  30 tests — API endpoint tests with
          │     Tests       │  real CSV fixtures, real file I/O
          └────────┬────────┘
     ┌─────────────┴─────────────┐
     │        Unit Tests         │  200+ tests — calculators, validators,
     │                           │  repository logic, service rules
     └───────────────────────────┘
Coverage target: ≥ 80% across all layers
```

### Priority Test Areas

1. **Greeks calculations** — Black-Scholes delta, gamma, theta, vega against known reference values. Financial accuracy, not just code coverage.

2. **Manoeuvre snapshot** — Integration test: POST snapshot → read CSV → verify rows match payload. Test with NIFTY and BANKNIFTY simultaneously to catch instrument-month scoping regressions.

3. **CSV repository layer** — Unit test each repo's read/write/upsert. Test PK uniqueness checks, schema validation, concurrent write safety.

4. **System API contract tests** — Pact or OpenAPI-based consumer contract tests. Ensures BFF always gets what it expects from System APIs even as they evolve.

5. **Config validation** — Unit test that `ConfigManager` raises on invalid env vars (negative port, missing required secrets in production mode).

### Example Integration Test

```python
# tests/integration/test_manoeuvre_snapshot.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_snapshot_scoped_to_instrument(tmp_path, client: AsyncClient):
    """NIFTY and BANKNIFTY snapshots are written independently."""
    nifty_payload = make_snapshot_payload(und="NIFTY", month="APR", pnl=10000)
    bn_payload    = make_snapshot_payload(und="BANKNIFTY", month="APR", pnl=-5000)

    r1 = await client.post("/api/v1/manoeuvre/snapshot", json=nifty_payload)
    r2 = await client.post("/api/v1/manoeuvre/snapshot", json=bn_payload)
    assert r1.status_code == 200
    assert r2.status_code == 200

    # Read history for NIFTY only
    r3 = await client.get("/api/v1/snapshots/pnl-history?und=NIFTY&month=APR")
    days = r3.json()["days"]
    # Should not contain BANKNIFTY rows
    for day in days:
        for g in day["groups"]:
            assert g.get("und", "NIFTY") == "NIFTY"
```

---

## 9. Configuration Management

### Current State

`src/rita/config.py` — 83 lines of `os.getenv()` calls with no validation, no type coercion, no required-field enforcement. CORS is `*`. Port is an unvalidated string.

### Target — Validated Settings

```python
# src/rita/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Required in production
    portfolio_api_key: str = Field(min_length=32)
    jwt_secret: str        = Field(min_length=32)

    # Deployment
    environment: str       = Field(default="development",
                                   pattern=r"^(development|staging|production)$")
    api_port: int          = Field(default=8000, ge=1, le=65535)
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])

    # Paths
    output_dir: str = Field(default="./rita_output")
    input_dir: str  = Field(default="./rita_input")

    # Feature flags
    use_database: bool = False      # False in v1, True in v2

    @validator("portfolio_api_key")
    def api_key_not_default(cls, v, values):
        if values.get("environment") == "production" and v in ("dev", "test", ""):
            raise ValueError("Must set a strong PORTFOLIO_API_KEY in production")
        return v

settings = Settings()  # fails fast at import if config is invalid
```

### Environment-Specific Config

```
config/
  base.yaml          # shared defaults
  development.yaml   # local overrides (CORS=*, debug logging)
  staging.yaml       # staging-specific (restricted CORS, staging DB)
  production.yaml    # production (no debug, strict CORS, secrets from vault)
```

---

## 10. Financial Domain Specifics

### Greeks Calculation Testing

`portfolio_manager.py` has 946 lines including Black-Scholes Greeks. These are used for margin calculations. They are currently untested. In production with real capital this is a P0 risk.

```python
# tests/unit/test_greeks.py
import pytest
from rita.core.portfolio_manager import black_scholes_greeks

# Reference values from options pricing textbooks
@pytest.mark.parametrize("spot,strike,ttm,opt_type,expected_delta", [
    (22700, 22700, 0.05, "CE", 0.50),   # ATM call delta ≈ 0.50
    (22700, 22700, 0.05, "PE", -0.50),  # ATM put delta ≈ -0.50
    (22700, 23000, 0.05, "CE", 0.28),   # OTM call
    (22700, 22400, 0.05, "PE", -0.28),  # OTM put
])
def test_delta(spot, strike, ttm, opt_type, expected_delta):
    delta, gamma, theta, vega = black_scholes_greeks(spot, strike, ttm, opt_type)
    assert abs(delta - expected_delta) < 0.02  # within 2% of reference
```

### Lot Size Management

Hardcoded `LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30}` in `portfolio_manager.py` line 28. NSE revises lot sizes. This must be config-driven and validated against the actual position quantities in the portfolio:

```python
# rita_input/lot_sizes.csv  ← new normalised table
# und, lot_size, effective_from, effective_to
# NIFTY, 65, 2024-07-01,
# NIFTY, 75, 2021-01-01, 2024-06-30
# BANKNIFTY, 30, 2023-01-01,
```

### Audit Trail

Every change to a position, scenario level, or model activation must be recorded:

```python
# rita_output/audit_log.csv  ← new table
# audit_id, ts, table_name, record_id, action, old_values_json, new_values_json, user
```

In v2 this maps directly to a PostgreSQL `audit_log` table with triggers.

---

## 11. Frontend Architecture

### Current State

| File | Size | Problem |
|------|------|---------|
| `dashboard/rita.html` | 142 KB | Single-file SPA — 4,000+ lines of HTML + CSS + JS |
| `dashboard/fno.html` | 134 KB | Same — 3,500+ lines |
| `dashboard/ops.html` | 75 KB | Same |
| `dashboard/fno-manoeuvre.js` | 35 KB | Extracted helper, but still global state |

All pages are desktop-only. No responsive breakpoints exist. Layout breaks below ~1100px. Not usable on tablet or mobile.

---

### Hardcoded API URL (fix immediately)

```javascript
// dashboard/index.html — breaks in every non-local deployment
const API_BASE = 'http://localhost:8000';
```

Fix:
```javascript
// Resolves to relative path in production (same origin); explicit in dev
const API_BASE = window.RITA_API_BASE || window.location.origin;
```

In Kubernetes, inject at nginx serve time via a `config.js` sidecar or a `window.__RITA_CONFIG__` block in the HTML template.

---

### UI Section Decomposition

Each monolithic HTML file is decomposed into discrete sections. Every section is an independently loadable ES module with its own JS file, its own CSS file, and its own BFF endpoint. Sections can be loaded lazily — the shell page loads the nav and first visible section immediately; remaining sections load on tab activation.

#### `fno.html` — FnO Portfolio App

```
fno/
  index.html                    # Shell: topbar, sidebar nav, section mount points
  sections/
    portfolio-grid/
      portfolio-grid.js         # Positions table + Greeks columns
      portfolio-grid.css
      # BFF: GET /bff/fno/portfolio-grid
    risk-greeks/
      risk-greeks.js            # Delta/Gamma/Theta/Vega summary cards + curves
      risk-greeks.css
      # BFF: GET /bff/fno/risk-greeks
    hedge-radar/
      hedge-radar.js            # Radar chart + hedge quality scores
      hedge-radar.css
      # BFF: GET /bff/fno/hedge-radar
    hedge-history/
      hedge-history.js          # Historical hedge ratio timeline
      hedge-history.css
      # BFF: GET /bff/fno/hedge-history
    manoeuvre/
      manoeuvre.js              # Month tiles + group tabs + pool
      manoeuvre-groups.js       # Group card + sparklines (was fno-manoeuvre.js)
      manoeuvre-pool.js         # Position pool + drag-drop
      manoeuvre.css
      # BFF: GET /bff/fno/manoeuvre
    payoff/
      payoff.js                 # Strategy payoff curves at expiry
      payoff.css
      # BFF: GET /bff/fno/payoff
  shared/
    api-client.js               # All fetch() wrappers for fno app
    fno-formatters.js           # fmtPnl(), pnlClass(), fmtGreek()
    fno-state.js                # Shared state: positions[], scenarioLevels{}, marketData{}
```

#### `rita.html` — RL Model App

```
rita/
  index.html                    # Shell: topbar, sidebar nav
  sections/
    model-status/
      model-status.js           # Active model card + training history
      model-status.css
      # BFF: GET /bff/dashboard/model-status
    backtest/
      backtest.js               # Equity curve chart + metrics table
      backtest.css
      # BFF: GET /bff/dashboard/backtest
    trade-diagnostics/
      trade-diagnostics.js      # Per-trade win/loss analysis
      trade-diagnostics.css
      # BFF: GET /bff/dashboard/trade-diagnostics
    performance/
      performance.js            # Sharpe, MDD, CAGR, monthly heatmap
      performance.css
      # BFF: GET /bff/dashboard/performance
    workflow/
      workflow.js               # 8-step pipeline runner + step status
      workflow.css
      # BFF: POST /api/v1/workflow/run (Business Process API)
  shared/
    api-client.js
    rita-charts.js              # Chart.js wrapper (equity curve, heatmap)
```

#### `ops.html` — Operations Portal App

```
ops/
  index.html                    # Shell: topbar, sidebar nav
  sections/
    daily-ops/
      daily-ops.js              # Snapshot status cards + trigger buttons
      daily-ops.css
      # BFF: GET /bff/ops/daily-status
    session-notes/
      session-notes.js          # Session note history timeline
      session-notes.css
      # BFF: GET /bff/ops/session-notes
    action-log/
      action-log.js             # Drag-drop action log table + filters
      action-log.css
      # BFF: GET /bff/ops/action-log
    chat-analytics/
      chat-analytics.js         # Chat classifier analytics
      chat-analytics.css
      # BFF: GET /bff/ops/chat-analytics
  shared/
    api-client.js
```

#### Shared Design System (across all apps)

```
shared/
  design-system.css             # CSS custom properties, typography, spacing
  components/
    kpi-card.js                 # Reusable KPI tile (value + label + sub)
    badge.js                    # Status badges (ok/warn/danger/neutral)
    data-table.js               # Sortable, filterable table component
    sparkline.js                # Chart.js sparkline wrapper
    topbar.js                   # App shell topbar
    sidebar-nav.js              # Collapsible sidebar navigation
  config.js                     # API_BASE, auth headers, environment
  auth.js                       # Token management, logout
  error-handler.js              # Global fetch error + toast notifications
```

**Section loading pattern:**
```javascript
// index.html shell — lazy loads sections on nav click
async function loadSection(sectionId) {
  const mount = document.getElementById('section-mount');
  if (sectionCache[sectionId]) {
    mount.innerHTML = sectionCache[sectionId];
    return;
  }
  const { render } = await import(`./sections/${sectionId}/${sectionId}.js`);
  sectionCache[sectionId] = await render(mount);
}
```

---

### Responsive Design Strategy

RITA is primarily a desktop trading tool, but read-only views must work on tablet for monitoring during market hours, and critical alerts must be readable on mobile. The strategy uses a **progressive enhancement** model: full functionality on desktop, read-only monitoring on tablet, summary/alert view on mobile.

#### Breakpoints

```css
/* shared/design-system.css */
:root {
  --bp-mobile:  480px;   /* iPhone SE and up */
  --bp-tablet:  768px;   /* iPad portrait and up */
  --bp-desktop: 1100px;  /* Standard laptop and up */
  --bp-wide:    1440px;  /* Wide monitor */
}

/* Usage */
@media (max-width: 767px)  { /* mobile */  }
@media (max-width: 1099px) { /* tablet */  }
@media (min-width: 1440px) { /* wide */    }
```

#### Layout Behaviour Per Breakpoint

| Element | Desktop (≥1100px) | Tablet (768–1099px) | Mobile (<768px) |
|---------|-------------------|---------------------|-----------------|
| Shell sidebar | 220px fixed | Collapsible overlay (hamburger) | Hidden — bottom tab bar |
| FnO Portfolio Grid | All columns visible | Hide Greeks columns; show on row expand | Card-per-position layout |
| Greeks cards | 4-column row | 2-column grid | Single column stack |
| Manoeuvre month tiles | 6 tiles in one row | 3 tiles per row | 2 tiles per row |
| Group tab strip | Horizontal pill tabs | Horizontal scroll | Dropdown select |
| Manoeuvre group card | Table with 5 columns | 3 columns; SL/Target on expand | Card list |
| Position Pool | 2-column grid | 1-column grid | 1-column compact |
| Backtest equity chart | Full width | Full width | Full width, no legend |
| KPI strip | 4–6 per row | 3 per row | 2 per row, smaller font |
| Topbar | Logo + nav links + user | Logo + hamburger | Logo + hamburger |
| Snapshot history table | 4 columns | 3 columns | 2 columns |

#### Sidebar Collapse (tablet)

The sidebar becomes an overlay on tablet rather than taking up a fixed column, maximising the content area:

```css
/* shared/sidebar-nav.js CSS */
@media (max-width: 1099px) {
  .shell { grid-template-columns: 0px 1fr; }  /* already in ops.html */

  .sidebar {
    position: fixed;
    left: -220px;
    top: 0; bottom: 0;
    z-index: 300;
    transition: left .22s cubic-bezier(.4,0,.2,1);
    box-shadow: 4px 0 16px rgba(0,0,0,.35);
  }
  .sidebar.open { left: 0; }

  .overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,.45);
    z-index: 299;
  }
  .overlay.visible { display: block; }
}
```

#### Mobile Bottom Tab Bar (replaces sidebar)

On mobile, the sidebar is replaced by a bottom tab bar with 4–5 primary destinations per app:

```
FnO app tabs:    Portfolio | Greeks | Manoeuvre | Hedge | Ops
Rita app tabs:   Backtest  | Model  | Trades    | Workflow
Ops app tabs:    Daily Ops | Notes  | Actions   | Chat
```

```css
@media (max-width: 767px) {
  .sidebar { display: none; }

  .bottom-tab-bar {
    display: flex;
    position: fixed; bottom: 0; left: 0; right: 0;
    height: 56px;
    background: var(--surface);
    border-top: 1.5px solid var(--border);
    z-index: 200;
  }

  .bottom-tab-bar button {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 2px;
    font-size: 9px; font-weight: 600;
    color: var(--t3);
    background: none; border: none;
    cursor: pointer;
  }

  .bottom-tab-bar button.active { color: var(--p01); }

  /* Give content area bottom padding so it clears the tab bar */
  .content { padding-bottom: 64px; }
}
```

#### FnO Portfolio Grid — Responsive Columns

The position table collapses gracefully rather than horizontal-scrolling on narrow screens:

```javascript
// sections/portfolio-grid/portfolio-grid.js
const COLUMNS = {
  desktop: ['instrument', 'side', 'qty', 'avg', 'ltp', 'pnl',
            'delta', 'gamma', 'theta', 'vega', 'margin'],
  tablet:  ['instrument', 'side', 'qty', 'pnl', 'delta', 'margin'],
  mobile:  ['instrument', 'pnl'],   // expandable row reveals full detail
};

function getVisibleColumns() {
  const w = window.innerWidth;
  if (w >= 1100) return COLUMNS.desktop;
  if (w >= 768)  return COLUMNS.tablet;
  return COLUMNS.mobile;
}

// Mobile: tapping a row expands an inline detail card
function renderMobileRow(position) {
  return `
    <div class="pos-card" onclick="toggleDetail(this)">
      <div class="pos-card-header">
        <span class="pos-instrument">${position.instrument}</span>
        <span class="pnl ${pnlClass(position.pnl)}">${fmtPnl(position.pnl)}</span>
      </div>
      <div class="pos-card-detail" hidden>
        <div class="detail-row"><span>Qty</span><span>${position.qty}</span></div>
        <div class="detail-row"><span>Avg</span><span>${position.avg}</span></div>
        <div class="detail-row"><span>Delta</span><span>${position.delta?.toFixed(2)}</span></div>
        <div class="detail-row"><span>Margin</span><span>${fmtInr(position.margin)}</span></div>
      </div>
    </div>`;
}
```

#### Manoeuvre Section — Responsive

```css
/* Tablets: 2-column group layout instead of side-by-side group+spark */
@media (max-width: 1099px) {
  .man-group-layout { grid-template-columns: 1fr; }
  .man-spark-panel  { min-height: 120px; }
  .man-month-grid   { grid-template-columns: repeat(3, 1fr); }
}

/* Mobile: month tiles 2-up, tabs become a dropdown */
@media (max-width: 767px) {
  .man-month-grid { grid-template-columns: repeat(2, 1fr); }
  .man-tab-strip  { display: none; }
  .man-tab-select { display: block; width: 100%; }  /* <select> fallback */
}
```

#### Touch Interactions

On tablet/mobile, drag-and-drop (used in Manoeuvre pool) must be supplemented with a tap-to-assign flow, since touch drag-drop is unreliable across browsers:

```javascript
// manoeuvre-pool.js — tap-to-assign for touch devices
function isTouchDevice() {
  return window.matchMedia('(pointer: coarse)').matches;
}

if (isTouchDevice()) {
  // Tap pool item → highlight it as "selected"
  // Then tap a group tab → assign to that group
  // No drag-drop; avoids touch event complications
  poolEl.addEventListener('click', e => {
    const lotEl = e.target.closest('[data-lot-key]');
    if (lotEl) selectLotForAssignment(lotEl.dataset.lotKey);
  });
  groupTabsEl.addEventListener('click', e => {
    const tab = e.target.closest('[data-group-id]');
    if (tab && selectedLotKey) assignLot(selectedLotKey, tab.dataset.groupId);
  });
}
```

#### Font and Density

Trading UIs are information-dense on desktop. On mobile, density must be reduced to avoid mis-taps:

```css
@media (max-width: 767px) {
  :root {
    --fd: 'Inter', sans-serif;
    font-size: 14px;              /* up from 13px */
    --cell-padding: 10px 12px;   /* up from 6px 8px */
    --touch-target: 44px;        /* min tap target per WCAG */
  }

  button, .btn-sm, .man-tab {
    min-height: var(--touch-target);
  }
}
```

#### Read-Only Mode on Mobile

Certain operations (drag-drop group assignment, scenario level edits, snapshot save) are write operations that require precise interaction. On mobile, these are hidden and the user sees a read-only view with a "Open on desktop to edit" banner:

```javascript
const READ_ONLY_SECTIONS = ['manoeuvre-pool', 'scenario-editor'];

if (window.innerWidth < 768) {
  READ_ONLY_SECTIONS.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.setAttribute('inert', '');
      el.insertAdjacentHTML('beforebegin',
        '<div class="mobile-read-only-banner">View only — open on desktop to edit</div>');
    }
  });
}
```

#### Target Frontend Folder Structure (v1)

```
frontend/                                 # replaces dashboard/
  shared/
    design-system.css                     # CSS variables, typography, spacing
    components/
      kpi-card.js
      badge.js
      data-table.js
      sparkline.js
      topbar.js
      sidebar-nav.js
      bottom-tab-bar.js                   # mobile only
    config.js                             # API_BASE, environment
    auth.js
    error-handler.js
    responsive.js                         # getBreakpoint(), isTouchDevice()
  apps/
    fno/
      index.html
      sections/
        portfolio-grid/  ...
        risk-greeks/     ...
        hedge-radar/     ...
        hedge-history/   ...
        manoeuvre/       ...
        payoff/          ...
      shared/
        fno-state.js
        fno-formatters.js
        api-client.js
    rita/
      index.html
      sections/
        model-status/    ...
        backtest/        ...
        trade-diagnostics/ ...
        performance/     ...
        workflow/        ...
      shared/
        rita-charts.js
        api-client.js
    ops/
      index.html
      sections/
        daily-ops/       ...
        session-notes/   ...
        action-log/      ...
        chat-analytics/  ...
      shared/
        api-client.js
    index/
      index.html                          # Landing page
```

**v1 uses native ES modules** (`<script type="module">`) — no build step required. Works in all modern browsers without webpack/vite.

**v2 migrates to React + TypeScript + Vite** — each app in `apps/` becomes a React SPA, each section becomes a `<Section />` component, shared components become a proper component library with Storybook, and all responsive behaviour is unit-tested with `@testing-library/react`.

---

## 12. Summary Priority Table

| Priority | Category | Change | Est. Effort |
|----------|----------|--------|-------------|
| **P0** | Security | Replace hardcoded API key with JWT + secure cookie | 3 days |
| **P0** | Security | Restrict CORS to known origins; add rate limiting | 1 day |
| **P0** | API Design | Implement three-tier API structure (BFF / BP / System) | 3 weeks |
| **P0** | Data | Normalise all CSVs to match DB table schema | 1 week |
| **P0** | Testing | Greek calculation unit tests against reference values | 3 days |
| **P0** | Testing | Integration tests for manoeuvre snapshot (instrument-month scoping) | 2 days |
| **P0** | Logging | Replace print() with structlog JSON structured logging | 2 days |
| **P0** | Deployment | Kubernetes manifests + health endpoints + shared volume | 1 week |
| **P0** | CI/CD | GitHub Actions pipeline: lint → test → build → deploy | 3 days |
| **P1** | Data | CSV repository layer with schema validation and PK checks | 1 week |
| **P1** | Config | Pydantic Settings with validated required fields | 2 days |
| **P1** | Error handling | Global exception handler + trace IDs + NaN logging | 2 days |
| **P1** | Testing | 80% unit + integration test coverage | 3 weeks |
| **P1** | Deployment | Secrets in cloud vault (not env vars) | 2 days |
| **P1** | Resiliency | Circuit breaker + timeouts + graceful shutdown | 3 days |
| **P1** | Observability | Prometheus metrics + AlertManager rules | 3 days |
| **P1** | Frontend | Decompose HTML apps into section modules per app | 2 weeks |
| **P1** | Frontend | Shared design system CSS + shared component JS files | 3 days |
| **P1** | Frontend | Fix hardcoded API URL; inject API_BASE via config.js | 1 day |
| **P1** | Responsive | Breakpoints + sidebar overlay + bottom tab bar (mobile) | 1 week |
| **P1** | Responsive | FnO portfolio grid responsive columns + mobile card layout | 3 days |
| **P1** | Responsive | Manoeuvre responsive tiles + tab→dropdown on mobile | 2 days |
| **P1** | Responsive | Tap-to-assign flow for touch devices (replaces drag-drop) | 2 days |
| **P2** | Lot sizes | Move LOT_SIZES to lot_sizes.csv with effective dates | 1 day |
| **P2** | Model | model_registry.csv with version metadata and is_active flag | 2 days |
| **P2** | Audit | audit_log.csv for all portfolio/scenario level changes | 2 days |
| **v2** | Data | PostgreSQL migration (mechanical COPY from normalised CSVs) | 3–4 weeks |
| **v2** | Frontend | React + TypeScript + Vite; component tests; Storybook | 4 weeks |
| **v2** | Responsive | Full responsive test suite with @testing-library/react | 1 week |

---

## 13. Conclusion

RITA is production-viable with a focused v1 effort. The three architectural pillars defined here — CSV-as-normalised-tables, three-tier API, and cloud-native deployment — are designed so that each subsequent version builds on the previous without rework:

- **v1 establishes the shape.** The API tier boundaries, table schemas, service interfaces, and cloud infrastructure are all defined and deployed. CSVs are just a storage implementation detail.
- **v2 swaps the storage.** PostgreSQL replaces CSVs by switching a dependency-injected repository implementation. No business logic changes.
- **v3 adds scale and intelligence.** Real-time streaming, ML-driven manoeuvre suggestions, multi-user RBAC — all on a foundation that already handles concurrency, auth, and observability.

The highest-risk gap today is the **API monolith** — 1,533 lines of mixed concerns make every change risky and every test hard to write. This is the first thing to address in v1, because everything else (testing, security, CI/CD) depends on having clean, independently testable units.

> **Key rule:** The CSV schema IS the database schema. Every column name added or changed in a CSV must match what will become a SQL column name in v2. Enforce this in code review from day one.
