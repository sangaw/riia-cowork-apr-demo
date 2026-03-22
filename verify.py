"""Quick verification script for RITA Phase 1"""
import sys
sys.path.insert(0, "src")

import os
CSV = os.environ.get("NIFTY_CSV_PATH", "")

print("=== Test 1: Data Loader ===")
from rita.core.data_loader import load_nifty_csv, get_historical_stats, get_training_data, get_validation_data, get_backtest_data
df = load_nifty_csv(CSV)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
stats = get_historical_stats(df)
print(f"CAGR: {stats['cagr_pct']}%, Sharpe: {stats['sharpe_ratio']}, Max DD: {stats['max_drawdown_pct']}%")
print(f"Best year: {stats['best_year_pct']}%, Worst year: {stats['worst_year_pct']}%")
train = get_training_data(df)
val = get_validation_data(df)
bt = get_backtest_data(df)
print(f"Train: {len(train)} rows, Val: {len(val)} rows, Backtest: {len(bt)} rows")
print("PASS: data_loader\n")

print("=== Test 2: Technical Analyzer ===")
from rita.core.technical_analyzer import calculate_indicators, get_market_summary
feat_df = calculate_indicators(df)
print(f"Features added: {[c for c in feat_df.columns if c not in df.columns]}")
summary = get_market_summary(feat_df)
print(f"Trend: {summary['trend']}, RSI: {summary['rsi_14']}, Sentiment: {summary['sentiment_proxy']}")
print("PASS: technical_analyzer\n")

print("=== Test 3: Goal Engine ===")
from rita.core.goal_engine import set_goal
goal = set_goal(15.0, 365, "moderate", stats)
print(f"Feasibility: {goal['feasibility']}")
print(f"Note: {goal['feasibility_note']}")
print("PASS: goal_engine\n")

print("=== Test 4: Strategy Engine ===")
from rita.core.strategy_engine import design_strategy
strategy = design_strategy(summary, goal)
print(f"Strategy: {strategy['name']}")
print(f"Allocation: {strategy['base_allocation_pct']}%")
print("PASS: strategy_engine\n")

print("=== Test 5: Python Client (steps 1-3 only, skip RL training) ===")
from rita.interfaces.python_client import RITAClient
client = RITAClient(CSV, output_dir="./rita_output_test")
r1 = client.set_goal(15.0, 365, "moderate")
print(f"Step 1: {r1['result']['feasibility']}")
r2 = client.analyze_market()
print(f"Step 2: trend={r2['result']['trend']}")
r3 = client.design_strategy()
print(f"Step 3: {r3['result']['name']}")
print("PASS: python_client (steps 1-3)\n")

print("All verification tests PASSED!")
print("Note: RL training (step 4) skipped in quick test — run separately as it takes ~5-10 min.")
