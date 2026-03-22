"""
RITA Orchestration — Workflow Orchestrator
8-step pipeline runner. Composes all core modules.
All methods return structured dicts suitable for any caller (MCP, Python client, REST API).
"""

import json
import os

import numpy as np
import pandas as pd

from rita.core.data_loader import (
    load_nifty_csv,
    get_training_data,
    get_validation_data,
    get_backtest_data,
    get_bear_episodes,
    get_historical_stats,
    BACKTEST_START,
)
from rita.core.technical_analyzer import calculate_indicators, get_market_summary, detect_regime
from rita.core.rl_agent import (
    train_agent, train_best_of_n, train_bear_model, load_agent,
    run_episode, run_regime_episode, validate_agent,
)
from rita.core.performance import generate_full_report
from rita.core.goal_engine import set_goal, update_goal_from_results
from rita.core.strategy_engine import design_strategy, validate_strategy_constraints
from rita.core.risk_engine import RiskEngine
from rita.core.training_tracker import TrainingTracker
from rita.core.shap_explainer import SHAPExplainer
from rita.orchestration.session import SessionManager
from rita.orchestration.monitor import PhaseMonitor
from rita.config import TRAIN_TIMESTEPS, BEAR_TRAIN_TIMESTEPS


# ─── Episode cache helpers ────────────────────────────────────────────────────

def _model_mtime(model_path: str) -> float:
    """Return the model file's modification timestamp, or 0.0 if file missing."""
    try:
        return os.path.getmtime(model_path)
    except OSError:
        return 0.0


def _save_episode_cache(episode: dict, path: str, model_path: str = "") -> None:
    """Serialise a run_episode() result dict to JSON (DataFrames excluded).

    Embeds the model's mtime so _load_episode_cache can detect staleness.
    """
    obs = episode.get("observations")
    cacheable = {
        "_model_mtime": _model_mtime(model_path),
        "portfolio_values": episode["portfolio_values"],
        "benchmark_values": episode["benchmark_values"],
        "allocations": episode["allocations"],
        "daily_returns": episode["daily_returns"],
        "dates": [str(d) for d in episode["dates"]],
        "close_prices": episode["close_prices"],
        "q_confidence_series": episode.get("q_confidence_series", []),
        "observations": obs.tolist() if obs is not None else [],
        "performance": {k: (v if not isinstance(v, (np.bool_, np.integer, np.floating)) else v.item())
                        for k, v in episode["performance"].items()},
    }
    # Atomic write: write to temp file then rename to avoid partial-write corruption
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(cacheable, fh)
    os.replace(tmp_path, path)


def _load_episode_cache(path: str, model_path: str = "") -> dict | None:
    """Load a cached episode dict from JSON, or return None if missing or stale.

    Stale = the model file has been modified since the cache was written.
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)

    # Invalidate if model has changed since cache was written
    if model_path:
        cached_mtime = data.get("_model_mtime", 0.0)
        current_mtime = _model_mtime(model_path)
        if current_mtime > cached_mtime + 1:  # 1-second tolerance for filesystem precision
            print(f"[RITA] Cache stale (model updated since cache written): {os.path.basename(path)}")
            return None

    data["dates"] = pd.DatetimeIndex(data["dates"])
    if data.get("observations"):
        data["observations"] = np.array(data["observations"], dtype=np.float32)
    return data


class WorkflowOrchestrator:
    """
    Orchestrates the 8-step RITA investment workflow.

    Each step method:
      - starts/ends monitoring
      - calls core module(s)
      - stores result in session
      - returns a plain dict (no protocol-specific types)
    """

    def __init__(self, csv_path: str, output_dir: str = "./rita_output"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.session = SessionManager(output_dir)
        self.session.load()   # restore persisted state across API restarts
        self.monitor = PhaseMonitor(output_dir)

        # Load raw + feature-enriched data once
        self._raw_df: pd.DataFrame = None
        self._feat_df: pd.DataFrame = None

    def _ensure_data(self) -> None:
        if self._raw_df is None:
            self._raw_df = load_nifty_csv(self.csv_path)
            self._feat_df = calculate_indicators(self._raw_df)

    # ─── STEP 1 ──────────────────────────────────────────────────────────────

    def step1_set_goal(
        self,
        target_return_pct: float,
        time_horizon_days: int,
        risk_tolerance: str,
    ) -> dict:
        """Step 1: Set financial goal anchored to real Nifty 50 history."""
        self.monitor.start_step(1)
        try:
            self._ensure_data()
            hist_stats = get_historical_stats(self._raw_df)
            goal = set_goal(target_return_pct, time_horizon_days, risk_tolerance, hist_stats)
            self.session.set("goal", goal)
            self.session.set("historical_stats", hist_stats)
            self.session.save()
            self.monitor.end_step(1, {"feasibility": goal["feasibility"],
                                      "target_pct": target_return_pct})
            return {"step": 1, "name": "Set Financial Goal", "result": goal}
        except Exception as e:
            self.monitor.fail_step(1, str(e))
            raise

    # ─── STEP 2 ──────────────────────────────────────────────────────────────

    def step2_analyze_market(self) -> dict:
        """Step 2: Analyze current market conditions using last 252 trading days."""
        self.monitor.start_step(2)
        try:
            self._ensure_data()
            # Use the last 252 rows of feature data as "current" market window
            recent = self._feat_df.iloc[-252:]
            summary = get_market_summary(recent)
            self.session.set("research", summary)
            self.session.save()
            self.monitor.end_step(2, {"trend": summary["trend"], "rsi": summary["rsi_14"]})
            return {"step": 2, "name": "Analyze Market Conditions", "result": summary}
        except Exception as e:
            self.monitor.fail_step(2, str(e))
            raise

    # ─── STEP 3 ──────────────────────────────────────────────────────────────

    def step3_design_strategy(self) -> dict:
        """Step 3: Design strategy based on research + goal, check constraints."""
        self.monitor.start_step(3)
        try:
            research = self.session.get("research")
            goal = self.session.get("goal")
            if research is None or goal is None:
                raise RuntimeError("Run step1 and step2 before step3.")
            strategy = design_strategy(research, goal)
            self.session.set("strategy", strategy)
            self.session.save()
            self.monitor.end_step(3, {"strategy": strategy["name"]})
            return {"step": 3, "name": "Design Strategy", "result": strategy}
        except Exception as e:
            self.monitor.fail_step(3, str(e))
            raise

    # ─── STEP 4 ──────────────────────────────────────────────────────────────

    def step4_train_model(
        self,
        timesteps: int = TRAIN_TIMESTEPS,
        force_retrain: bool = False,
        n_seeds: int = 1,
        model_type: str = "bull",
    ) -> dict:
        """
        Step 4: Train DDQN on 2010-2022, validate on 2023-2024.

        model_type:
            "bull"  — train/load bull model (9-feature, default). Uses full 2010-2022 training set.
            "bear"  — train bear specialist model. Extracts correction episodes from 2010-2022,
                      uses 300k timesteps and bear reward function. Saved as rita_ddqn_bear_model.zip.
            "both"  — train bull model then bear model sequentially.

        force_retrain=False: reuse existing model file if present.
        n_seeds>1: best-of-N seed selection.
        """
        self.monitor.start_step(4)
        try:
            self._ensure_data()

            if model_type == "bear":
                return self._step4_train_bear(force_retrain=force_retrain, n_seeds=n_seeds)
            elif model_type == "both":
                bull_result = self._step4_train_bull(
                    timesteps=timesteps, force_retrain=force_retrain, n_seeds=n_seeds
                )
                bear_result = self._step4_train_bear(force_retrain=force_retrain, n_seeds=n_seeds)
                combined = {
                    "step": 4, "name": "Train DDQN Models (Bull + Bear)",
                    "result": {"bull": bull_result["result"], "bear": bear_result["result"]},
                }
                self.monitor.end_step(4, {"model_type": "both"})
                return combined
            else:
                return self._step4_train_bull(
                    timesteps=timesteps, force_retrain=force_retrain, n_seeds=n_seeds
                )
        except Exception as e:
            self.monitor.fail_step(4, str(e))
            raise

    def _step4_train_bull(self, timesteps: int = TRAIN_TIMESTEPS, force_retrain: bool = False, n_seeds: int = 1) -> dict:
        """Train or load the bull model (rita_ddqn_model.zip)."""
        model_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")

        # Reuse existing model if available and not forced to retrain
        if not force_retrain and os.path.exists(model_zip):
            model = load_agent(model_zip)
            val_df = calculate_indicators(get_validation_data(self._raw_df))
            val_metrics = validate_agent(model, val_df)

            self.session.set("model_path", model_zip)
            self.session.set("validation_metrics", val_metrics)
            self.session.set("model_source", "loaded_existing")
            self.session.save()

            self._cache_episodes_for_risk_view(model)

            result = {"model_path": model_zip, "source": "loaded_existing", "validation": val_metrics}
            self.monitor.end_step(4, {
                "source": "loaded_existing",
                "sharpe_validation": val_metrics["sharpe_ratio"],
                "mdd_validation": val_metrics["max_drawdown_pct"],
            })
            return {"step": 4, "name": "Train DDQN Model", "result": result}

        # Train from scratch
        train_df = calculate_indicators(get_training_data(self._raw_df))
        val_df = calculate_indicators(get_validation_data(self._raw_df))

        if n_seeds > 1:
            model, training_metrics = train_best_of_n(
                train_df, val_df, self.output_dir, timesteps=timesteps, n_seeds=n_seeds
            )
        else:
            model, training_metrics = train_agent(train_df, self.output_dir, timesteps=timesteps)
        val_metrics = validate_agent(model, val_df)

        self.session.set("model_path", training_metrics["model_path"])
        self.session.set("training_metrics", training_metrics)
        self.session.set("validation_metrics", val_metrics)
        self.session.set("model_source", "trained")
        self.session.save()

        self._cache_episodes_for_risk_view(model)

        result = {**training_metrics, "source": "trained", "validation": val_metrics}
        self.monitor.end_step(4, {
            "source": "trained",
            "sharpe_validation": val_metrics["sharpe_ratio"],
            "mdd_validation": val_metrics["max_drawdown_pct"],
        })
        return {"step": 4, "name": "Train DDQN Model", "result": result}

    def _step4_train_bear(self, force_retrain: bool = False, n_seeds: int = 5) -> dict:
        """Train or load the bear specialist model (rita_ddqn_bear_model.zip).

        Also ensures model_path (bull model) is set in the session so that
        step5 and step6 can run after a bear-only training.
        """
        bear_zip = os.path.join(self.output_dir, "rita_ddqn_bear_model.zip")
        bull_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")

        # Ensure bull model is registered in session — step6 needs it for regime backtest
        if not self.session.get("model_path"):
            if os.path.exists(bull_zip):
                val_df = calculate_indicators(get_validation_data(self._raw_df))
                bull_val = validate_agent(load_agent(bull_zip), val_df)
                self.session.set("model_path", bull_zip)
                self.session.set("validation_metrics", bull_val)
                self.session.set("model_source", "loaded_existing")
            else:
                raise RuntimeError(
                    "No bull model found at rita_ddqn_model.zip. "
                    "Train the bull model first (model_type='bull') before training the bear model."
                )

        if not force_retrain and os.path.exists(bear_zip):
            model = load_agent(bear_zip)
            val_df = calculate_indicators(get_validation_data(self._raw_df))
            val_metrics = validate_agent(model, val_df)
            self.session.set("bear_model_path", bear_zip)
            self.session.set("bear_validation_metrics", val_metrics)
            self.session.save()
            result = {"model_path": bear_zip, "source": "loaded_existing", "validation": val_metrics}
            self.monitor.end_step(4, {
                "source": "loaded_existing_bear",
                "sharpe_validation": val_metrics["sharpe_ratio"],
            })
            return {"step": 4, "name": "Train Bear Model", "result": result}

        # Extract bear episodes from training data and train
        full_train_df = calculate_indicators(get_training_data(self._raw_df))
        bear_df = get_bear_episodes(full_train_df)
        val_df = calculate_indicators(get_validation_data(self._raw_df))

        # Bear model: simpler policy (mostly cash) → fewer seeds + fewer timesteps
        # Cap at 3 seeds regardless of UI setting — bear data is small (~600 rows)
        bear_seeds = min(n_seeds, 3)
        print(f"[RITA] Bear episodes: {len(bear_df)} rows from {len(full_train_df)} training rows (seeds={bear_seeds})")
        model, training_metrics = train_bear_model(
            bear_df, val_df, self.output_dir, timesteps=BEAR_TRAIN_TIMESTEPS, n_seeds=bear_seeds
        )
        val_metrics = validate_agent(model, val_df)

        self.session.set("bear_model_path", training_metrics["model_path"])
        self.session.set("bear_training_metrics", training_metrics)
        self.session.set("bear_validation_metrics", val_metrics)
        self.session.save()

        result = {**training_metrics, "source": "trained", "validation": val_metrics}
        self.monitor.end_step(4, {
            "source": "trained_bear",
            "bear_episodes_rows": len(bear_df),
            "sharpe_validation": val_metrics["sharpe_ratio"],
        })
        return {"step": 4, "name": "Train Bear Model", "result": result}

    # ─── STEP 5 ──────────────────────────────────────────────────────────────

    def step5_set_simulation_period(
        self,
        start: str = BACKTEST_START,
        end: str = None,
    ) -> dict:
        """Step 5: Set the backtest simulation period (default: 2025 data)."""
        self.monitor.start_step(5)
        try:
            self._ensure_data()
            if end is None:
                end = self._raw_df.index.max().strftime("%Y-%m-%d")

            # Validate period doesn't overlap training data
            if start < "2023-01-01":
                raise ValueError(
                    f"Simulation start {start} overlaps training period (2010-2022). "
                    "Use 2023-01-01 or later."
                )

            sim_period = {"start": start, "end": end}
            self.session.set("simulation_period", sim_period)
            self.session.save()
            self.monitor.end_step(5, sim_period)
            return {"step": 5, "name": "Set Simulation Period", "result": sim_period}
        except Exception as e:
            self.monitor.fail_step(5, str(e))
            raise

    # ─── STEP 6 ──────────────────────────────────────────────────────────────

    def step6_run_backtest(self, backtest_mode: str = "auto") -> dict:
        """
        Step 6: Run trained DDQN model on the simulation period.

        backtest_mode:
            "auto"   — regime-aware if bear model file exists, otherwise bull only
            "bull"   — always use bull model only (ignore bear model)
            "regime" — force regime switching (raises if no bear model)
        """
        self.monitor.start_step(6)
        try:
            model_path = self.session.get("model_path")
            # Fall back to the model file on disk (survives API restarts)
            if not model_path:
                default_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")
                if os.path.exists(default_zip):
                    model_path = default_zip
                    self.session.set("model_path", model_path)
            sim_period = self.session.get("simulation_period")
            if not model_path or not sim_period:
                raise RuntimeError("Run step4 and step5 before step6.")

            self._ensure_data()
            # Fetch 200 extra warmup days before start so indicators (MACD, EMA, ATR)
            # have enough history to be non-NaN by the time the slice begins.
            WARMUP_DAYS = 200
            start_ts = pd.Timestamp(sim_period["start"])
            warmup_start = (start_ts - pd.offsets.BDay(WARMUP_DAYS)).strftime("%Y-%m-%d")
            raw_with_warmup = get_backtest_data(self._raw_df, warmup_start, sim_period["end"])
            full_indicators = calculate_indicators(raw_with_warmup)
            # Trim back to the requested period after indicators are computed
            backtest_df = full_indicators[full_indicators.index >= start_ts]

            bull_model = load_agent(model_path)

            # Resolve bear model based on backtest_mode
            bear_model = None
            if backtest_mode != "bull":
                bear_zip = os.path.join(self.output_dir, "rita_ddqn_bear_model.zip")
                bear_model_path = self.session.get("bear_model_path") or (bear_zip if os.path.exists(bear_zip) else None)
                if bear_model_path and os.path.exists(bear_model_path):
                    bear_model = load_agent(bear_model_path)
                elif backtest_mode == "regime":
                    raise RuntimeError(
                        "backtest_mode='regime' requires a trained bear model. "
                        "Train it first with model_type='bear'."
                    )

            use_regime = bear_model is not None
            if use_regime:
                print(f"[RITA] Regime-aware backtest (mode={backtest_mode})")
                backtest_results = run_regime_episode(bull_model, backtest_df, bear_model=bear_model)
                regime_info = detect_regime(backtest_df)
                backtest_results["regime_info"] = regime_info
                n_bear = sum(1 for r in backtest_results.get("regime_series", []) if r == "BEAR")
                print(f"[RITA] Regime split: {n_bear} BEAR days / {len(backtest_results['allocations']) - n_bear} BULL days")
            else:
                print(f"[RITA] Bull-only backtest (mode={backtest_mode})")
                backtest_results = run_episode(bull_model, backtest_df)
                backtest_results["regime_series"] = ["BULL"] * len(backtest_results["allocations"])

            self.session.set("backtest_results", backtest_results)
            self.session.save()

            perf = backtest_results["performance"]
            self.monitor.end_step(6, {
                "sharpe": perf["sharpe_ratio"],
                "mdd": perf["max_drawdown_pct"],
                "return": perf["portfolio_total_return_pct"],
                "backtest_mode": backtest_mode,
                "regime_aware": use_regime,
            })
            return {
                "step": 6,
                "name": "Run Backtest",
                "result": {
                    "performance": perf,
                    "days_simulated": perf["total_days"],
                    "backtest_mode": backtest_mode,
                    "regime_aware": use_regime,
                }
            }
        except Exception as e:
            self.monitor.fail_step(6, str(e))
            raise

    # ─── STEP 7 ──────────────────────────────────────────────────────────────

    def step7_get_results(self, record_to_history: bool = True, notes: str = "") -> dict:
        """Step 7: Generate full performance report with interpretability plots."""
        self.monitor.start_step(7)
        try:
            backtest_results = self.session.get("backtest_results")
            if not backtest_results:
                raise RuntimeError("Run step6 before step7.")

            plots = generate_full_report(
                portfolio_values=np.array(backtest_results["portfolio_values"]),
                benchmark_values=np.array(backtest_results["benchmark_values"]),
                allocations=backtest_results["allocations"],
                dates=backtest_results["dates"],
                close_prices=np.array(backtest_results["close_prices"]),
                q_values_by_feature=backtest_results.get("q_values_by_feature"),
                output_dir=self.output_dir,
            )

            constraint_check = validate_strategy_constraints(backtest_results)
            perf = backtest_results["performance"]

            # ── Risk Arc ─────────────────────────────────────────────────────
            try:
                self._compute_risk_arc()
            except Exception as re:
                print(f"[RITA] Risk arc computation skipped: {re}")

            # ── SHAP Explainability ───────────────────────────────────────────
            try:
                self._compute_shap_explainability()
            except Exception as se:
                print(f"[RITA] SHAP computation skipped: {se}")

            # ── Training History ──────────────────────────────────────────────
            if record_to_history:
                try:
                    # Inject trade count into perf so training_tracker can store it
                    allocations = backtest_results.get("allocations", [])
                    if allocations:
                        import pandas as _pd
                        _alloc = _pd.Series(allocations)
                        perf["total_trades"] = int((_alloc.diff().fillna(0).abs() > 0).sum())
                    self._record_training_round(perf, notes)
                except Exception as te:
                    print(f"[RITA] Training history recording skipped: {te}")

            result = {
                "performance": perf,
                "constraint_check": constraint_check,
                "plots": plots,
                "output_dir": self.output_dir,
            }
            self.monitor.end_step(7, {
                "plots_generated": len(plots),
                "constraints_met": constraint_check["all_constraints_met"],
            })
            return {"step": 7, "name": "Get Results", "result": result}
        except Exception as e:
            self.monitor.fail_step(7, str(e))
            raise

    # ─── STEP 8 ──────────────────────────────────────────────────────────────

    def step8_update_goal(self) -> dict:
        """Step 8: Update financial goal based on backtest results (closed loop)."""
        self.monitor.start_step(8)
        try:
            original_goal = self.session.get("goal")
            backtest_results = self.session.get("backtest_results")
            if not original_goal or not backtest_results:
                raise RuntimeError("Run steps 1–7 before step8.")

            updated_goal = update_goal_from_results(original_goal, backtest_results)
            self.session.set("updated_goal", updated_goal)
            self.session.save()
            self.monitor.end_step(8, {
                "assessment": updated_goal["assessment"],
                "revised_target_pct": updated_goal["revised_target_pct"],
            })
            return {"step": 8, "name": "Update Financial Goal", "result": updated_goal}
        except Exception as e:
            self.monitor.fail_step(8, str(e))
            raise

    # ─── Risk Arc & Training Tracker (internal) ───────────────────────────────

    def _cache_episodes_for_risk_view(self, model) -> None:
        """
        Run model inference on training + validation data and cache results to disk.
        Called from step4 so that step7 can build the full risk arc without
        rerunning inference a second time.
        """
        self._ensure_data()
        train_cache = os.path.join(self.output_dir, "train_episode_cache.json")
        val_cache = os.path.join(self.output_dir, "val_episode_cache.json")

        train_df = calculate_indicators(get_training_data(self._raw_df))
        val_df = calculate_indicators(get_validation_data(self._raw_df))

        model_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")
        print("[RITA] Caching train episode for Risk View (may take a few seconds)…")
        train_ep = run_episode(model, train_df)
        _save_episode_cache(train_ep, train_cache, model_path=model_zip)

        print("[RITA] Caching val episode for Risk View…")
        val_ep = run_episode(model, val_df)
        _save_episode_cache(val_ep, val_cache, model_path=model_zip)

    def _compute_risk_arc(self) -> None:
        """
        Build the full-arc risk timeline (Train → Validation → Backtest) and
        persist results to risk_timeline.csv, risk_trade_events.csv,
        risk_summary.json inside output_dir.
        Called from step7_get_results().
        """
        backtest_results = self.session.get("backtest_results")
        if not backtest_results:
            return

        self._ensure_data()
        nifty_returns = self._feat_df["daily_return"].dropna()
        risk_engine = RiskEngine()

        sim_period = self.session.get("simulation_period") or {}
        bt_start = sim_period.get("start", BACKTEST_START)
        bt_end = sim_period.get("end")

        # Feature DataFrames for regime detection per phase
        train_feat = calculate_indicators(get_training_data(self._raw_df))
        val_feat = calculate_indicators(get_validation_data(self._raw_df))
        bt_feat = calculate_indicators(get_backtest_data(self._raw_df, bt_start, bt_end))

        # Load cached episodes (produced in step4) — invalidated automatically if model changed
        model_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")
        train_ep = _load_episode_cache(
            os.path.join(self.output_dir, "train_episode_cache.json"), model_path=model_zip
        )
        val_ep = _load_episode_cache(
            os.path.join(self.output_dir, "val_episode_cache.json"), model_path=model_zip
        )

        # Fall back: rerun inference if cache is missing or stale, then re-save
        if train_ep is None:
            model_path = self.session.get("model_path")
            if model_path and os.path.exists(model_path):
                print("[RITA] Risk arc: cache missing/stale — rerunning inference to refresh…")
                m = load_agent(model_path)
                train_ep = run_episode(m, train_feat)
                val_ep = run_episode(m, val_feat)
                _save_episode_cache(
                    train_ep,
                    os.path.join(self.output_dir, "train_episode_cache.json"),
                    model_path=model_zip,
                )
                _save_episode_cache(
                    val_ep,
                    os.path.join(self.output_dir, "val_episode_cache.json"),
                    model_path=model_zip,
                )

        timelines = []

        if train_ep is not None:
            tl_train = risk_engine.compute_risk_timeline(
                train_ep, train_feat, nifty_returns, phase="Train"
            )
            timelines.append(tl_train)

        if val_ep is not None:
            tl_val = risk_engine.compute_risk_timeline(
                val_ep, val_feat, nifty_returns, phase="Validation"
            )
            timelines.append(tl_val)

        tl_bt = risk_engine.compute_risk_timeline(
            backtest_results, bt_feat, nifty_returns, phase="Backtest"
        )
        timelines.append(tl_bt)

        combined = risk_engine.combine_phases(timelines)
        trades = risk_engine.compute_trade_events(combined)
        summary = risk_engine.compute_risk_summary(combined, trades)

        risk_engine.save(combined, trades, summary, self.output_dir)
        self.session.set("risk_arc_ready", True)
        self.session.save()

    def _compute_shap_explainability(self) -> None:
        """
        Fit SHAP DeepExplainer on training observations (background) and
        explain the backtest observations. Saves shap_values.npz,
        shap_importance.csv, shap_phase_importance.csv to output_dir.
        """
        model_path = self.session.get("model_path")
        backtest_results = self.session.get("backtest_results")
        if not model_path or not backtest_results:
            return

        # Background observations — from cached training episode (invalidated if model changed)
        model_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")
        train_ep = _load_episode_cache(
            os.path.join(self.output_dir, "train_episode_cache.json"), model_path=model_zip
        )
        if train_ep is None or not len(train_ep.get("observations", [])):
            print("[RITA] SHAP: train episode cache missing — skipping")
            return

        train_obs = np.array(train_ep["observations"], dtype=np.float32)

        # Explain observations — backtest + validation episodes combined
        bt_obs = backtest_results.get("observations")
        if bt_obs is None or len(bt_obs) == 0:
            print("[RITA] SHAP: backtest observations missing — skipping")
            return

        val_ep = _load_episode_cache(
            os.path.join(self.output_dir, "val_episode_cache.json"), model_path=model_zip
        )

        obs_parts, phase_parts = [bt_obs], ["Backtest"] * len(bt_obs)
        if val_ep is not None and len(val_ep.get("observations", [])):
            val_obs = np.array(val_ep["observations"], dtype=np.float32)
            obs_parts.insert(0, val_obs)
            phase_parts = ["Validation"] * len(val_obs) + phase_parts

        explain_obs = np.concatenate(obs_parts, axis=0)
        phases = np.array(phase_parts)

        model = load_agent(model_path)
        explainer = SHAPExplainer(model)

        print("[RITA] Fitting SHAP explainer (background: training obs)…")
        explainer.fit(train_obs)

        print(f"[RITA] Computing SHAP values for {len(explain_obs)} observations…")
        shap_vals = explainer.explain(explain_obs)

        explainer.save(shap_vals, explain_obs, phases, self.output_dir)
        print("[RITA] SHAP values saved.")

    def _record_training_round(self, backtest_perf: dict, notes: str = "") -> None:
        """Append one row to training_history.csv via TrainingTracker."""
        val_metrics = self.session.get("validation_metrics") or {}
        training_metrics = self.session.get("training_metrics") or {
            "source": self.session.get("model_source", "loaded_existing"),
            "timesteps_trained": 0,
        }
        tracker = TrainingTracker(self.output_dir)
        rn = tracker.record_round(training_metrics, val_metrics, backtest_perf, notes)
        print(f"[RITA] Training history: recorded Round {rn}")

    # ─── FULL PIPELINE ────────────────────────────────────────────────────────

    def run_pipeline(self, config: dict) -> dict:
        """
        Run all 8 steps sequentially. Useful for automated pipeline runs.

        config keys:
            target_return_pct   (float)
            time_horizon_days   (int)
            risk_tolerance      (str: conservative/moderate/aggressive)
            timesteps           (int, optional, default 200_000)
            sim_start           (str, optional, default BACKTEST_START)
            sim_end             (str, optional)
        """
        results = {}
        results["step1"] = self.step1_set_goal(
            config["target_return_pct"],
            config["time_horizon_days"],
            config["risk_tolerance"],
        )
        results["step2"] = self.step2_analyze_market()
        results["step3"] = self.step3_design_strategy()
        results["step4"] = self.step4_train_model(
            timesteps=config.get("timesteps", TRAIN_TIMESTEPS)
        )
        results["step5"] = self.step5_set_simulation_period(
            start=config.get("sim_start", BACKTEST_START),
            end=config.get("sim_end"),
        )
        results["step6"] = self.step6_run_backtest()
        results["step7"] = self.step7_get_results()
        results["step8"] = self.step8_update_goal()

        results["progress"] = self.monitor.get_progress_report()
        return results
