"""
RITA Orchestration — Workflow Orchestrator
8-step pipeline runner. Composes all core modules.
All methods return structured dicts suitable for any caller (MCP, Python client, REST API).
"""

import os

import pandas as pd

from rita.core.data_loader import (
    load_nifty_csv,
    get_training_data,
    get_validation_data,
    get_backtest_data,
    get_historical_stats,
    BACKTEST_START,
)
from rita.core.technical_analyzer import calculate_indicators, get_market_summary
from rita.core.rl_agent import train_agent, load_agent, run_episode, validate_agent
from rita.core.performance import generate_full_report, compute_all_metrics
from rita.core.goal_engine import set_goal, update_goal_from_results
from rita.core.strategy_engine import design_strategy, validate_strategy_constraints
from rita.orchestration.session import SessionManager
from rita.orchestration.monitor import PhaseMonitor


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

    def step4_train_model(self, timesteps: int = 200_000, force_retrain: bool = False) -> dict:
        """
        Step 4: Train DDQN on 2010-2022, validate on 2023-2024.

        If a trained model already exists at output_dir/rita_ddqn_model.zip and
        force_retrain=False, the existing model is loaded instead of retraining.
        Pass force_retrain=True to always retrain from scratch.
        """
        self.monitor.start_step(4)
        try:
            model_zip = os.path.join(self.output_dir, "rita_ddqn_model.zip")
            existing_path = self.session.get("model_path")

            # Reuse existing model if available and not forced to retrain
            if not force_retrain and os.path.exists(model_zip):
                model = load_agent(model_zip)
                self._ensure_data()
                val_df = calculate_indicators(get_validation_data(self._raw_df))
                val_metrics = validate_agent(model, val_df)

                self.session.set("model_path", model_zip)
                self.session.set("validation_metrics", val_metrics)
                self.session.save()

                result = {
                    "model_path": model_zip,
                    "source": "loaded_existing",
                    "validation": val_metrics,
                }
                self.monitor.end_step(4, {
                    "source": "loaded_existing",
                    "sharpe_validation": val_metrics["sharpe_ratio"],
                    "mdd_validation": val_metrics["max_drawdown_pct"],
                })
                return {"step": 4, "name": "Train DDQN Model", "result": result}

            # Train from scratch
            self._ensure_data()
            train_df = calculate_indicators(get_training_data(self._raw_df))
            val_df = calculate_indicators(get_validation_data(self._raw_df))

            model, training_metrics = train_agent(train_df, self.output_dir, timesteps=timesteps)
            val_metrics = validate_agent(model, val_df)

            self.session.set("model_path", training_metrics["model_path"])
            self.session.set("training_metrics", training_metrics)
            self.session.set("validation_metrics", val_metrics)
            self.session.save()

            result = {**training_metrics, "source": "trained", "validation": val_metrics}
            self.monitor.end_step(4, {
                "source": "trained",
                "sharpe_validation": val_metrics["sharpe_ratio"],
                "mdd_validation": val_metrics["max_drawdown_pct"],
            })
            return {"step": 4, "name": "Train DDQN Model", "result": result}
        except Exception as e:
            self.monitor.fail_step(4, str(e))
            raise

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

    def step6_run_backtest(self) -> dict:
        """Step 6: Run trained DDQN model on the simulation period."""
        self.monitor.start_step(6)
        try:
            model_path = self.session.get("model_path")
            sim_period = self.session.get("simulation_period")
            if not model_path or not sim_period:
                raise RuntimeError("Run step4 and step5 before step6.")

            model = load_agent(model_path)
            self._ensure_data()
            backtest_df = calculate_indicators(
                get_backtest_data(self._raw_df, sim_period["start"], sim_period["end"])
            )

            backtest_results = run_episode(model, backtest_df)
            self.session.set("backtest_results", backtest_results)
            self.session.save()

            perf = backtest_results["performance"]
            self.monitor.end_step(6, {
                "sharpe": perf["sharpe_ratio"],
                "mdd": perf["max_drawdown_pct"],
                "return": perf["portfolio_total_return_pct"],
            })
            return {
                "step": 6,
                "name": "Run Backtest",
                "result": {
                    "performance": perf,
                    "days_simulated": perf["total_days"],
                }
            }
        except Exception as e:
            self.monitor.fail_step(6, str(e))
            raise

    # ─── STEP 7 ──────────────────────────────────────────────────────────────

    def step7_get_results(self) -> dict:
        """Step 7: Generate full performance report with interpretability plots."""
        self.monitor.start_step(7)
        try:
            backtest_results = self.session.get("backtest_results")
            if not backtest_results:
                raise RuntimeError("Run step6 before step7.")

            import numpy as np
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
            timesteps=config.get("timesteps", 200_000)
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
