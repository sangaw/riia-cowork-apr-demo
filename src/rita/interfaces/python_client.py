"""
RITA Interface — Python Client
Direct Python API that wraps WorkflowOrchestrator.
No HTTP, no MCP — just a clean Python class.

Usage:
    from rita.interfaces.python_client import RITAClient

    client = RITAClient("C:/path/to/merged.csv")
    client.set_goal(15.0, 365, "moderate")
    client.analyze_market()
    client.design_strategy()
    client.train_model()              # trains on 2010-2022, validates 2023-2024
    client.set_simulation_period()   # defaults to 2025
    client.run_backtest()
    results = client.get_results()
    client.update_goal()
"""

import os

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import BACKTEST_START
from rita.config import TRAIN_TIMESTEPS


class RITAClient:
    """
    Clean Python API to the RITA 8-step workflow.
    All methods return plain dicts.
    """

    def __init__(self, csv_path: str, output_dir: str = "./rita_output"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Nifty CSV not found: {csv_path}")
        self.orchestrator = WorkflowOrchestrator(csv_path, output_dir)

    def set_goal(
        self,
        target_return_pct: float,
        time_horizon_days: int,
        risk_tolerance: str = "moderate",
    ) -> dict:
        """Step 1: Set financial goal."""
        return self.orchestrator.step1_set_goal(
            target_return_pct, time_horizon_days, risk_tolerance
        )

    def analyze_market(self) -> dict:
        """Step 2: Analyze current market conditions."""
        return self.orchestrator.step2_analyze_market()

    def design_strategy(self) -> dict:
        """Step 3: Design strategy based on research and goal."""
        return self.orchestrator.step3_design_strategy()

    def train_model(self, timesteps: int = TRAIN_TIMESTEPS, force_retrain: bool = False) -> dict:
        """Step 4: Train DDQN on 2010-2022, validate on 2023-2024.
        Reuses existing model unless force_retrain=True."""
        return self.orchestrator.step4_train_model(timesteps=timesteps, force_retrain=force_retrain)

    def set_simulation_period(
        self, start: str = BACKTEST_START, end: str = None
    ) -> dict:
        """Step 5: Set the backtest period (default: 2025)."""
        return self.orchestrator.step5_set_simulation_period(start, end)

    def run_backtest(self) -> dict:
        """Step 6: Run the trained model on the simulation period."""
        return self.orchestrator.step6_run_backtest()

    def get_results(self) -> dict:
        """Step 7: Get full results report and generate interpretability plots."""
        return self.orchestrator.step7_get_results()

    def update_goal(self) -> dict:
        """Step 8: Update financial goal based on backtest results (feedback loop)."""
        return self.orchestrator.step8_update_goal()

    def run_full_pipeline(self, config: dict) -> dict:
        """
        Run all 8 steps sequentially.

        config keys:
            target_return_pct   (float)
            time_horizon_days   (int)
            risk_tolerance      (str)
            timesteps           (int, optional)
            sim_start           (str, optional)
            sim_end             (str, optional)
        """
        return self.orchestrator.run_pipeline(config)

    def get_progress(self) -> dict:
        """Return a summary of completed steps."""
        return self.orchestrator.session.get_progress_summary()
