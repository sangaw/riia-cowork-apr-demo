"""
RITA Interface — Streamlit Web UI (Phase 2)
Interactive dashboard: 8-step pipeline, live progress, Plotly charts, HTML export.

Run with:
    streamlit run src/rita/interfaces/streamlit_app.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from rita.orchestration.workflow import WorkflowOrchestrator
from rita.core.data_loader import BACKTEST_START
from rita.core.performance import RISK_FREE_RATE, TRADING_DAYS

# ─── Constants ────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CSV_PATH = os.getenv(
    "NIFTY_CSV_PATH",
    r"C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\raw-data\nifty\merged.csv",
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./rita_output")

STEP_NAMES = [
    "Set Goal", "Analyze Market", "Design Strategy", "Train Model",
    "Set Period", "Run Backtest", "Get Results", "Update Goal",
]

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RITA — Nifty 50 RL Investment System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.title("RITA")
    st.sidebar.caption("Nifty 50 · Double DQN · 8-Step Workflow")

    # Data source toggle
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Source",
        ["Historical (Nifty 50)", "Mock Data (Coming Soon)"],
        index=0,
        label_visibility="collapsed",
    )
    if "Mock" in data_source:
        st.sidebar.info("Mock data support planned for Phase 3.")

    st.sidebar.divider()

    # Financial goal
    st.sidebar.subheader("Financial Goal")
    target_return = st.sidebar.slider("Target Annual Return (%)", 5.0, 30.0, 15.0, 0.5)
    horizon_days = st.sidebar.selectbox(
        "Time Horizon",
        [90, 180, 252, 365, 730],
        index=3,
        format_func=lambda x: f"{x} days",
    )
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1
    )

    st.sidebar.divider()

    # Model training
    st.sidebar.subheader("Model Training")
    model_zip = os.path.join(OUTPUT_DIR, "rita_ddqn_model.zip")
    model_exists = os.path.exists(model_zip)

    if model_exists:
        size_kb = round(os.path.getsize(model_zip) / 1024)
        st.sidebar.success(f"✓ Trained model found ({size_kb} KB)")
        force_retrain = st.sidebar.checkbox(
            "Force retrain from scratch",
            value=False,
            help="Uncheck to reuse the existing model (faster). Check to train a new model.",
        )
    else:
        st.sidebar.info("No trained model found — will train on first run.")
        force_retrain = True

    timesteps = 200_000
    if force_retrain or not model_exists:
        timesteps = st.sidebar.select_slider(
            "Training timesteps",
            options=[50_000, 100_000, 200_000, 500_000],
            value=200_000,
            format_func=lambda x: f"{x:,}",
        )

    st.sidebar.divider()

    # Simulation period
    st.sidebar.subheader("Simulation Period")
    sim_start = st.sidebar.text_input("Start date (YYYY-MM-DD)", BACKTEST_START)
    sim_end = st.sidebar.text_input("End date (blank = latest)", "")

    return {
        "target_return": target_return,
        "horizon_days": horizon_days,
        "risk_tolerance": risk_tolerance,
        "force_retrain": force_retrain,
        "timesteps": timesteps,
        "sim_start": sim_start,
        "sim_end": sim_end or None,
    }


# ─── Step progress bar ────────────────────────────────────────────────────────

def render_step_progress(completed: set, active: int = 0):
    cols = st.columns(8)
    for i, (col, name) in enumerate(zip(cols, STEP_NAMES), start=1):
        if i in completed:
            col.success(f"**{i}** ✓")
        elif i == active:
            col.info(f"**{i}** ↻")
        else:
            col.markdown(f"**{i}**")
        col.caption(name)


# ─── Plotly charts ────────────────────────────────────────────────────────────

def chart_returns(portfolio_values, benchmark_values, dates) -> go.Figure:
    port = np.asarray(portfolio_values) / portfolio_values[0] * 100
    bench = np.asarray(benchmark_values) / benchmark_values[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(dates), y=port, name="DDQN Strategy",
                             line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=list(dates), y=bench, name="Nifty Buy & Hold",
                             line=dict(color="#FF9800", width=1.5, dash="dash")))
    fig.update_layout(
        title="Cumulative Returns (base = 100)",
        xaxis_title="Date", yaxis_title="Value (base 100)",
        height=360, hovermode="x unified", legend=dict(x=0, y=1),
        margin=dict(t=50, b=30),
    )
    return fig


def chart_drawdown(portfolio_values, dates) -> go.Figure:
    port = np.asarray(portfolio_values)
    running_max = np.maximum.accumulate(port)
    dd = (port - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=dd, fill="tozeroy", name="Drawdown",
        line=dict(color="#F44336", width=1),
        fillcolor="rgba(244,67,54,0.25)",
    ))
    fig.add_hline(y=-10, line_dash="dash", line_color="darkred",
                  annotation_text="-10% limit", annotation_position="bottom right")
    fig.update_layout(
        title="Portfolio Drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=300, margin=dict(t=50, b=30),
    )
    return fig


def chart_allocations(allocations, dates, close_prices) -> go.Figure:
    alloc_arr = np.asarray(allocations)
    alloc_dates = list(dates[1:] if len(dates) == len(alloc_arr) + 1 else dates)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Nifty 50 Close Price", "DDQN Allocation Decisions"],
        row_heights=[0.55, 0.45],
    )
    fig.add_trace(
        go.Scatter(x=list(dates), y=list(close_prices), name="Nifty Close",
                   line=dict(color="#555", width=1.2)),
        row=1, col=1,
    )

    color_map = {0.0: "#F44336", 0.5: "#FF9800", 1.0: "#4CAF50"}
    label_map = {0.0: "Cash (0%)", 0.5: "Half (50%)", 1.0: "Full (100%)"}
    for val in [0.0, 0.5, 1.0]:
        mask = alloc_arr == val
        fig.add_trace(
            go.Scatter(
                x=[alloc_dates[i] for i in range(len(alloc_dates)) if mask[i]],
                y=alloc_arr[mask] * 100,
                mode="markers", name=label_map[val],
                marker=dict(color=color_map[val], size=4, opacity=0.75),
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Allocation (%)", tickvals=[0, 50, 100], row=2, col=1)
    fig.update_layout(height=500, hovermode="x unified", margin=dict(t=50, b=30))
    return fig


def chart_rolling_sharpe(daily_returns, dates, window: int = 63) -> go.Figure:
    rets = pd.Series(daily_returns, index=pd.DatetimeIndex(dates[1:]))
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    roll = (
        (rets - daily_rf).rolling(window).mean()
        / rets.rolling(window).std()
        * math.sqrt(TRADING_DAYS)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll.values,
        name=f"{window}-day Sharpe", line=dict(color="#9C27B0", width=1.8),
        fill="tozeroy", fillcolor="rgba(156,39,176,0.08)",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                  annotation_text="Target 1.0", annotation_position="top right")
    fig.add_hline(y=0.0, line_color="gray", line_width=0.8)
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title="Date", yaxis_title="Sharpe Ratio",
        height=310, margin=dict(t=50, b=30),
    )
    return fig


def chart_q_values(q_values_by_feature: dict) -> go.Figure | None:
    features = [f for f in q_values_by_feature if q_values_by_feature[f]]
    if not features:
        return None

    action_colors = {
        "Cash (0%)": "#F44336",
        "Half (50%)": "#FF9800",
        "Full (100%)": "#4CAF50",
    }
    fig = make_subplots(rows=1, cols=len(features), subplot_titles=features)

    for col_idx, feat in enumerate(features, start=1):
        data = q_values_by_feature[feat]
        buckets = list(data.keys())
        for action, color in action_colors.items():
            vals = [float(np.mean(data[b].get(action, [0]))) for b in buckets]
            fig.add_trace(
                go.Bar(name=action, x=buckets, y=vals, marker_color=color,
                       opacity=0.85, showlegend=(col_idx == 1)),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title="DDQN Interpretability — Avg Q-values by Feature Bucket",
        barmode="group", height=360, margin=dict(t=60, b=30),
    )
    return fig


# ─── Metrics row ──────────────────────────────────────────────────────────────

def render_metrics(perf: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    alpha = perf["portfolio_total_return_pct"] - perf["benchmark_total_return_pct"]

    c1.metric("Portfolio Return", f"{perf['portfolio_total_return_pct']:.1f}%",
              delta=f"{alpha:+.1f}% vs BnH",
              delta_color="normal" if alpha >= 0 else "inverse")
    c2.metric("CAGR", f"{perf['portfolio_cagr_pct']:.1f}%",
              delta=f"BnH: {perf['benchmark_cagr_pct']:.1f}%", delta_color="off")
    c3.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.3f}",
              delta="✓ ≥ 1.0" if perf["sharpe_constraint_met"] else "✗ < 1.0",
              delta_color="normal" if perf["sharpe_constraint_met"] else "inverse")
    c4.metric("Max Drawdown", f"{perf['max_drawdown_pct']:.1f}%",
              delta="✓ < 10%" if perf["drawdown_constraint_met"] else "✗ Exceeded",
              delta_color="normal" if perf["drawdown_constraint_met"] else "inverse")
    c5.metric("Win Rate", f"{perf['win_rate_pct']:.1f}%",
              delta=f"{perf['total_days']} days", delta_color="off")


# ─── HTML report builder ──────────────────────────────────────────────────────

def build_html_report(summary: dict) -> str:
    perf = summary.get("performance", {})
    goal = summary.get("goal", {})
    updated = summary.get("updated_goal", {})
    period = summary.get("simulation_period", {})
    constraints = summary.get("constraint_check", {})

    rows = "\n".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td><b>{v}</b></td></tr>"
        for k, v in perf.items()
    )

    recs = ""
    for r in constraints.get("recommendations", []):
        recs += f"<li>{r}</li>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RITA Backtest Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 48px; color: #333; max-width: 900px; }}
  h1 {{ color: #1565C0; margin-bottom: 4px; }}
  h2 {{ color: #1976D2; border-bottom: 2px solid #E3F2FD; padding-bottom: 4px; margin-top: 36px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ padding: 8px 14px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #E3F2FD; font-weight: 600; }}
  tr:nth-child(even) {{ background: #FAFAFA; }}
  .ok {{ color: #2E7D32; font-weight: bold; }}
  .fail {{ color: #B71C1C; font-weight: bold; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.85em; }}
  .badge-green {{ background: #C8E6C9; color: #1B5E20; }}
  .badge-orange {{ background: #FFE0B2; color: #E65100; }}
  .badge-red {{ background: #FFCDD2; color: #B71C1C; }}
  footer {{ margin-top: 48px; font-size: 0.8em; color: #999; border-top: 1px solid #eee; padding-top: 12px; }}
</style>
</head><body>
<h1>RITA — Nifty 50 Backtest Report</h1>
<p>
  <b>Simulation Period:</b> {period.get('start', 'N/A')} → {period.get('end', 'N/A')} &nbsp;|&nbsp;
  <b>Goal:</b> {goal.get('target_return_pct', 'N/A')}% target &nbsp;|&nbsp;
  <b>Risk:</b> {goal.get('risk_tolerance', 'N/A')} &nbsp;|&nbsp;
  <b>Feasibility:</b> {goal.get('feasibility', 'N/A')}
</p>

<h2>Performance Summary</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>

<h2>Constraint Check</h2>
<table>
  <tr><th>Constraint</th><th>Value</th><th>Limit</th><th>Status</th></tr>
  <tr>
    <td>Sharpe Ratio</td>
    <td>{perf.get('sharpe_ratio', 'N/A')}</td>
    <td>&gt; 1.0</td>
    <td class="{'ok' if perf.get('sharpe_constraint_met') else 'fail'}">
      {'✓ MET' if perf.get('sharpe_constraint_met') else '✗ NOT MET'}
    </td>
  </tr>
  <tr>
    <td>Max Drawdown</td>
    <td>{perf.get('max_drawdown_pct', 'N/A')}%</td>
    <td>&lt; 10%</td>
    <td class="{'ok' if perf.get('drawdown_constraint_met') else 'fail'}">
      {'✓ MET' if perf.get('drawdown_constraint_met') else '✗ NOT MET'}
    </td>
  </tr>
</table>
{f'<ul>{recs}</ul>' if recs else ''}

<h2>Goal Update (Step 8)</h2>
<p>
  Assessment: <span class="badge badge-{'green' if updated.get('assessment') == 'met' else 'orange' if updated.get('assessment') == 'close' else 'red'}">
    {updated.get('assessment', 'N/A').upper()}
  </span>
</p>
<p>{updated.get('recommendation', '')}</p>
<p><b>Revised target:</b> {updated.get('revised_target_pct', 'N/A')}%</p>

<footer>Generated by RITA Phase 2 — Double DQN Nifty 50 Investment System</footer>
</body></html>"""


# ─── Results dashboard ────────────────────────────────────────────────────────

def render_dashboard(orch: WorkflowOrchestrator, step_results: dict):
    backtest = orch.session.get("backtest_results")
    if not backtest:
        return

    perf = backtest["performance"]
    constraint_check = step_results.get("step7", {}).get("result", {}).get("constraint_check", {})

    st.subheader("Results Dashboard")
    tab_overview, tab_charts, tab_details, tab_export = st.tabs(
        ["📊 Overview", "📈 Charts", "🔍 Step Details", "📥 Export"]
    )

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_overview:
        render_metrics(perf)
        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Constraint Status**")
            if perf["sharpe_constraint_met"]:
                st.success(f"✓ Sharpe {perf['sharpe_ratio']:.3f} ≥ 1.0")
            else:
                st.warning(f"⚠ Sharpe {perf['sharpe_ratio']:.3f} — below 1.0 target")
            if perf["drawdown_constraint_met"]:
                st.success(f"✓ Max drawdown {perf['max_drawdown_pct']:.1f}% within −10% limit")
            else:
                st.error(f"✗ Max drawdown {perf['max_drawdown_pct']:.1f}% exceeded −10% limit")

            if constraint_check.get("recommendations"):
                st.markdown("**Recommendations**")
                for rec in constraint_check["recommendations"]:
                    st.info(rec, icon="💡")

        with col_r:
            st.markdown("**Goal Update (Step 8)**")
            s8 = step_results.get("step8", {}).get("result", {})
            if s8:
                assess = s8.get("assessment", "")
                color = "green" if assess == "met" else "orange" if assess == "close" else "red"
                st.markdown(f"Assessment: :{color}[**{assess.upper()}**]")
                st.write(s8.get("recommendation", ""))
                st.metric("Revised target", f"{s8.get('revised_target_pct', 'N/A')}%")
                if s8.get("constraint_violations"):
                    for v in s8["constraint_violations"]:
                        st.warning(v)

    # ── Charts ────────────────────────────────────────────────────────────────
    with tab_charts:
        dates = backtest["dates"]
        port_vals = backtest["portfolio_values"]
        bench_vals = backtest["benchmark_values"]
        allocs = backtest["allocations"]
        closes = backtest["close_prices"]
        daily_rets = backtest["daily_returns"]
        q_vals = backtest.get("q_values_by_feature", {})

        st.plotly_chart(chart_returns(port_vals, bench_vals, dates), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_drawdown(port_vals, dates), use_container_width=True)
        with col2:
            st.plotly_chart(chart_rolling_sharpe(daily_rets, dates), use_container_width=True)

        st.plotly_chart(chart_allocations(allocs, dates, closes), use_container_width=True)

        if q_vals:
            fig_q = chart_q_values(q_vals)
            if fig_q:
                st.plotly_chart(fig_q, use_container_width=True)

    # ── Step details ──────────────────────────────────────────────────────────
    with tab_details:
        for key in [f"step{i}" for i in range(1, 9)]:
            if key in step_results:
                d = step_results[key]
                with st.expander(f"Step {d['step']}: {d['name']}"):
                    st.json(d.get("result", {}))

        monitor_log = os.path.join(OUTPUT_DIR, "monitor_log.csv")
        if os.path.exists(monitor_log):
            st.markdown("**Phase timing**")
            st.dataframe(pd.read_csv(monitor_log), use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("**Download Results**")

        summary = {
            "performance": perf,
            "constraint_check": constraint_check,
            "goal": orch.session.get("goal"),
            "updated_goal": orch.session.get("updated_goal"),
            "strategy": orch.session.get("strategy"),
            "simulation_period": orch.session.get("simulation_period"),
        }

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇ JSON Summary",
            data=json.dumps(summary, indent=2, default=str),
            file_name="rita_results.json",
            mime="application/json",
            use_container_width=True,
        )
        col_b.download_button(
            "⬇ HTML Report",
            data=build_html_report(summary),
            file_name="rita_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.markdown("**Files saved to `rita_output/`**")
        rows = []
        for fname in [
            "session_state.csv", "backtest_daily.csv", "performance_summary.csv",
            "goal_history.csv", "monitor_log.csv", "rita_ddqn_model.zip",
        ]:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                rows.append({"File": fname, "Size (KB)": round(os.path.getsize(fpath) / 1024, 1)})

        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        if os.path.exists(plots_dir):
            for fname in sorted(os.listdir(plots_dir)):
                fpath = os.path.join(plots_dir, fname)
                rows.append({"File": f"plots/{fname}", "Size (KB)": round(os.path.getsize(fpath) / 1024, 1)})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(orch: WorkflowOrchestrator, config: dict, progress_slot):
    results = {}
    completed = set()

    def tick(n):
        completed.add(n)
        with progress_slot.container():
            render_step_progress(completed, active=n + 1 if n < 8 else 0)

    with st.status("Step 1 — Setting financial goal...", expanded=False) as s:
        r = orch.step1_set_goal(config["target_return"], config["horizon_days"], config["risk_tolerance"])
        results["step1"] = r
        tick(1)
        s.update(label=f"Step 1 ✓ — Goal: {r['result']['feasibility']} ({r['result']['target_return_pct']}%)")

    with st.status("Step 2 — Analyzing market conditions...", expanded=False) as s:
        r = orch.step2_analyze_market()
        results["step2"] = r
        tick(2)
        s.update(label=f"Step 2 ✓ — Trend: {r['result']['trend']} · RSI: {r['result']['rsi_14']:.1f}")

    with st.status("Step 3 — Designing strategy...", expanded=False) as s:
        r = orch.step3_design_strategy()
        results["step3"] = r
        tick(3)
        s.update(label=f"Step 3 ✓ — {r['result']['name']} · {r['result']['base_allocation_pct']}% allocation")

    model_zip = os.path.join(OUTPUT_DIR, "rita_ddqn_model.zip")
    force = config["force_retrain"]
    step4_label = (
        f"Step 4 — Training DDQN ({config['timesteps']:,} timesteps) — may take several minutes..."
        if force or not os.path.exists(model_zip)
        else "Step 4 — Loading existing trained model..."
    )
    with st.status(step4_label, expanded=force or not os.path.exists(model_zip)) as s:
        if force or not os.path.exists(model_zip):
            st.write("Training in progress. Please wait...")
        r = orch.step4_train_model(timesteps=config["timesteps"], force_retrain=force)
        results["step4"] = r
        val = r["result"]["validation"]
        source = r["result"].get("source", "trained")
        tick(4)
        s.update(label=f"Step 4 ✓ — {'Loaded' if source == 'loaded_existing' else 'Trained'} · Sharpe: {val['sharpe_ratio']:.3f} · MDD: {val['max_drawdown_pct']:.1f}%")

    with st.status("Step 5 — Setting simulation period...", expanded=False) as s:
        r = orch.step5_set_simulation_period(config["sim_start"], config["sim_end"])
        results["step5"] = r
        tick(5)
        s.update(label=f"Step 5 ✓ — {r['result']['start']} → {r['result']['end']}")

    with st.status("Step 6 — Running backtest...", expanded=False) as s:
        r = orch.step6_run_backtest()
        results["step6"] = r
        perf = r["result"]["performance"]
        tick(6)
        s.update(label=f"Step 6 ✓ — Return: {perf['portfolio_total_return_pct']:.1f}% · Sharpe: {perf['sharpe_ratio']:.3f}")

    with st.status("Step 7 — Generating results & plots...", expanded=False) as s:
        r = orch.step7_get_results()
        results["step7"] = r
        tick(7)
        s.update(label=f"Step 7 ✓ — {len(r['result']['plots'])} plots generated")

    with st.status("Step 8 — Updating financial goal...", expanded=False) as s:
        r = orch.step8_update_goal()
        results["step8"] = r
        tick(8)
        s.update(label=f"Step 8 ✓ — Assessment: {r['result']['assessment']} · Revised: {r['result']['revised_target_pct']:.1f}%")

    return results, completed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("RITA — Nifty 50 RL Investment System")
    st.caption("Double DQN · Train 2010–2022 · Validate 2023–2024 · Backtest 2025")

    config = render_sidebar()

    # Initialise session state
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}
    if "completed_steps" not in st.session_state:
        st.session_state.completed_steps = set()
    if "orch" not in st.session_state:
        if not os.path.exists(CSV_PATH):
            st.error(f"**Data file not found:** `{CSV_PATH}`")
            st.info("Set `NIFTY_CSV_PATH` env var or update the path in the source.")
            st.stop()
        st.session_state.orch = WorkflowOrchestrator(CSV_PATH, OUTPUT_DIR)

    orch: WorkflowOrchestrator = st.session_state.orch

    # Action buttons
    btn_run, btn_reset = st.columns([3, 1])
    run_clicked = btn_run.button("▶  Run Pipeline", type="primary", use_container_width=True)
    reset_clicked = btn_reset.button("↺  Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.step_results = {}
        st.session_state.completed_steps = set()
        st.session_state.orch = WorkflowOrchestrator(CSV_PATH, OUTPUT_DIR)
        st.rerun()

    progress_slot = st.empty()

    # Show existing progress if any
    if st.session_state.completed_steps and not run_clicked:
        with progress_slot.container():
            render_step_progress(st.session_state.completed_steps)

    if run_clicked:
        st.session_state.step_results = {}
        st.session_state.completed_steps = set()
        st.session_state.orch = WorkflowOrchestrator(CSV_PATH, OUTPUT_DIR)
        orch = st.session_state.orch
        try:
            results, completed = run_pipeline(orch, config, progress_slot)
            st.session_state.step_results = results
            st.session_state.completed_steps = completed
            st.success("Pipeline complete! 8/8 steps finished.")
        except Exception as e:
            st.error(f"Pipeline error at step {max(st.session_state.completed_steps, default=0) + 1}: {e}")
            raise

    # Dashboard
    if st.session_state.step_results:
        st.divider()
        render_dashboard(orch, st.session_state.step_results)


if __name__ == "__main__":
    main()
