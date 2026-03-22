"""Full 8-step RITA pipeline — end-to-end test with DDQN training."""
import sys
sys.path.insert(0, "src")

import os
CSV = os.environ.get("NIFTY_CSV_PATH", "")

from rita.interfaces.python_client import RITAClient
import json

client = RITAClient(CSV, output_dir="./rita_output")

def show(step, result):
    print(f"\n{'='*60}")
    print(f"STEP {step}: {result.get('name', '')}")
    print(f"{'='*60}")
    print(json.dumps(result.get("result", result), indent=2, default=str))

print("\nRITA Full Pipeline — Double DQN End-to-End Test")
print("Training on 2010-2022, Validation on 2023-2024, Backtest on 2025")

show(1, client.set_goal(15.0, 365, "moderate"))
show(2, client.analyze_market())
show(3, client.design_strategy())

print("\n[Step 4] Training DDQN (200k timesteps) — this takes ~5-10 min...")
show(4, client.train_model(timesteps=200_000))

show(5, client.set_simulation_period())   # defaults to 2025
show(6, client.run_backtest())
show(7, client.get_results())
show(8, client.update_goal())

print("\n" + "="*60)
print("Pipeline complete. Output saved to ./rita_output/")
print("Plots saved to ./rita_output/plots/")
progress = client.get_progress()
print(f"Steps completed: {progress['steps_completed']}/8 ({progress['pct_complete']}%)")
