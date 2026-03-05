"""Test steps 5-8 using the already-trained model (skip retraining)."""
import sys
sys.path.insert(0, "src")

CSV = r"C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\raw-data\nifty\merged.csv"

from rita.interfaces.python_client import RITAClient
import json

client = RITAClient(CSV, output_dir="./rita_output")

def show(step, result):
    print(f"\n{'='*60}")
    print(f"STEP {step}: {result.get('name', '')}")
    print(f"{'='*60}")
    print(json.dumps(result.get("result", result), indent=2, default=str))

# Re-run steps 1-3 quickly to populate session state
print("Re-populating session (steps 1-3)...")
client.set_goal(15.0, 365, "moderate")
client.analyze_market()
client.design_strategy()

# Manually inject model path from previous training
client.orchestrator.session.set("model_path", "./rita_output/rita_ddqn_model.zip")
client.orchestrator.session.save()
print("Model path injected: ./rita_output/rita_ddqn_model.zip")

# Now run steps 5-8
show(5, client.set_simulation_period())
show(6, client.run_backtest())
show(7, client.get_results())
show(8, client.update_goal())

print("\n" + "="*60)
print("Steps 5-8 complete!")
progress = client.get_progress()
print(f"Steps completed: {progress['steps_completed']}/8 ({progress['pct_complete']}%)")
