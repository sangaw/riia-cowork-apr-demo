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
from rita.core.risk_engine import RiskEngine
from rita.core.training_tracker import TrainingTracker
from rita.core.shap_explainer import SHAPExplainer, FEATURE_NAMES, ACTION_NAMES

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

    st.sidebar.divider()

    # Training history
    st.sidebar.subheader("Training History")
    record_run = st.sidebar.checkbox(
        "Record this run to training history",
        value=True,
        help="Appends metrics to training_history.csv for the Training Progress tab.",
    )
    run_notes = st.sidebar.text_input(
        "Round notes (optional)",
        placeholder="e.g. lr=1e-4, 300k steps",
    )

    return {
        "target_return": target_return,
        "horizon_days": horizon_days,
        "risk_tolerance": risk_tolerance,
        "force_retrain": force_retrain,
        "timesteps": timesteps,
        "sim_start": sim_start,
        "sim_end": sim_end or None,
        "record_run": record_run,
        "run_notes": run_notes,
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


# ─── Step detail strip (Dashboard tab) ────────────────────────────────────────

_STEP_ICONS = ["🎯", "📊", "🧩", "🤖", "📅", "🔁", "📋", "🔄"]
_SEL_KEY = "_dashboard_step_selected"


def _render_step_detail(result: dict):
    """Render step result dict as formatted key-value rows + nested expanders."""
    if not isinstance(result, dict) or not result:
        st.json(result)
        return
    scalars = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
    nested  = {k: v for k, v in result.items() if isinstance(v, (dict, list))}

    if scalars:
        n = min(len(scalars), 4)
        cols = st.columns(n)
        for col, (k, v) in zip(cols, scalars.items()):
            col.metric(k.replace("_", " ").title(), str(v))
        # overflow scalars
        remaining = list(scalars.items())[n:]
        if remaining:
            for k, v in remaining:
                st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")

    for k, v in nested.items():
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            label += f"  ({len(v)} items)"
        with st.expander(label):
            if isinstance(v, list):
                for item in v:
                    st.write(f"• {item}")
            else:
                st.json(v)


def render_step_strip(step_results: dict, output_dir: str):
    """
    Single-row step status strip. Click any step button to reveal its
    details in a panel below; click again to collapse.
    """
    step_keys = [f"step{i}" for i in range(1, 9)]

    if _SEL_KEY not in st.session_state:
        st.session_state[_SEL_KEY] = None

    st.markdown("**Pipeline Steps** — click any step to inspect details")
    cols = st.columns(8)
    for idx, (col, key) in enumerate(zip(cols, step_keys)):
        name  = STEP_NAMES[idx]
        icon  = _STEP_ICONS[idx]
        is_done = key in step_results
        is_sel  = st.session_state[_SEL_KEY] == key
        with col:
            with st.container(border=True):
                # Status line
                status_html = (
                    f"<div style='text-align:center;font-size:1.3em'>{icon} ✅</div>"
                    if is_done else
                    f"<div style='text-align:center;font-size:1.3em;color:#bbb'>{icon} ⏳</div>"
                )
                st.markdown(status_html, unsafe_allow_html=True)
                st.caption(name)
                if is_done:
                    btn_type = "primary" if is_sel else "secondary"
                    if st.button(
                        "▲ Hide" if is_sel else "▼ View",
                        key=f"stepbtn_{key}",
                        use_container_width=True,
                        type=btn_type,
                    ):
                        st.session_state[_SEL_KEY] = None if is_sel else key

    # ── Detail panel ──────────────────────────────────────────────────────────
    selected = st.session_state.get(_SEL_KEY)
    if selected and selected in step_results:
        d = step_results[selected]
        st.divider()
        idx = int(selected.replace("step", "")) - 1
        st.markdown(
            f"{_STEP_ICONS[idx]} &nbsp; **Step {d['step']}: {d['name']}**",
            unsafe_allow_html=True,
        )
        _render_step_detail(d.get("result", {}))

    # ── Phase timing ──────────────────────────────────────────────────────────
    monitor_log = os.path.join(output_dir, "monitor_log.csv")
    if os.path.exists(monitor_log):
        st.divider()
        with st.expander("⏱ Phase timing"):
            st.dataframe(pd.read_csv(monitor_log), use_container_width=True)


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


# ─── Risk View charts ─────────────────────────────────────────────────────────

PHASE_COLORS = {"Train": "#1565C0", "Validation": "#6A1B9A", "Backtest": "#2E7D32"}
REGIME_COLORS = {"Bull": "rgba(76,175,80,0.15)", "Bear": "rgba(244,67,54,0.15)", "Sideways": "rgba(158,158,158,0.08)"}
ALLOC_COLORS = {0.0: "#F44336", 0.5: "#FF9800", 1.0: "#4CAF50"}


def chart_risk_evolution(combined: pd.DataFrame) -> go.Figure:
    """
    Hero chart — 3-row subplot showing the full risk arc across all phases.
      Row 1: Rolling 20d Vol (%) + Portfolio VaR 95% (dual y-axis)
      Row 2: Allocation step-function coloured by level
      Row 3: Model confidence (Q-value spread)
    Phase boundaries marked with vertical dashed lines.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[
            "Market Volatility & Portfolio VaR 95%",
            "Allocation Decision (0% / 50% / 100%)",
            "Model Confidence (Q-value Spread)",
        ],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.06,
    )

    # ── Row 1: Vol + VaR ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=combined.index, y=combined["rolling_vol_20d"],
            name="20d Vol (ann.%)", line=dict(color="#FF9800", width=1.5),
            hovertemplate="%{y:.2f}%",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=combined.index, y=combined["portfolio_var_95"],
            name="Portfolio VaR 95%", line=dict(color="#F44336", width=2),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.12)",
            hovertemplate="%{y:.2f}%",
        ),
        row=1, col=1, secondary_y=True,
    )

    # ── Row 2: Allocation step-function ───────────────────────────────────────
    for alloc_val, color in ALLOC_COLORS.items():
        label = {0.0: "Cash (0%)", 0.5: "Half (50%)", 1.0: "Full (100%)"}[alloc_val]
        mask = combined["allocation"] == alloc_val
        fig.add_trace(
            go.Scatter(
                x=combined.index[mask], y=combined.loc[mask, "allocation"] * 100,
                mode="markers", name=label,
                marker=dict(color=color, size=3, opacity=0.7),
                showlegend=True,
            ),
            row=2, col=1,
        )

    # ── Row 3: Model confidence ───────────────────────────────────────────────
    has_conf = combined["model_confidence"].notna().any()
    if has_conf:
        fig.add_trace(
            go.Scatter(
                x=combined.index, y=combined["model_confidence"],
                name="Confidence", line=dict(color="#9C27B0", width=1.2),
                fill="tozeroy", fillcolor="rgba(156,39,176,0.10)",
                hovertemplate="%{y:.3f}",
            ),
            row=3, col=1,
        )

    # ── Phase boundary vertical lines ─────────────────────────────────────────
    for phase_name in ["Validation", "Backtest"]:
        phase_rows = combined[combined["phase"] == phase_name]
        if not phase_rows.empty:
            boundary = phase_rows.index[0]
            for r in [1, 2, 3]:
                fig.add_vline(
                    x=boundary, line_dash="dash", line_color="gray",
                    line_width=1.2, row=r, col=1,
                )
            # Label only on row 1
            fig.add_annotation(
                x=boundary, y=1, yref="paper",
                text=f"<b>{phase_name}</b>", showarrow=False,
                font=dict(size=10, color=PHASE_COLORS.get(phase_name, "gray")),
                xanchor="left", yanchor="top",
            )

    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="VaR 95% (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Allocation (%)", tickvals=[0, 50, 100], row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=3, col=1)
    fig.update_layout(
        title="Full Arc Risk Evolution: Train → Validation → Backtest",
        height=560, hovermode="x unified",
        legend=dict(x=1.02, y=1, xanchor="left"),
        margin=dict(t=70, b=30, r=160),
    )
    return fig


def chart_drawdown_budget(combined: pd.DataFrame) -> go.Figure:
    """
    Area chart showing % of the 10% MDD budget consumed over time.
    Traffic-light zones: green (0-50%), amber (50-80%), red (80-100%).
    """
    fig = go.Figure()

    # Colour-coded filled area
    budget = combined["drawdown_budget_pct"]
    fig.add_trace(go.Scatter(
        x=combined.index, y=budget,
        fill="tozeroy", fillcolor="rgba(76,175,80,0.2)",
        line=dict(color="#2E7D32", width=1.5),
        name="Budget used (%)",
        hovertemplate="%{y:.1f}%",
    ))

    # Danger zones as horizontal shapes
    fig.add_hrect(y0=50, y1=80, fillcolor="rgba(255,152,0,0.12)", line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(244,67,54,0.12)", line_width=0)

    fig.add_hline(y=50, line_dash="dot", line_color="#FF9800", line_width=1,
                  annotation_text="50% warning", annotation_position="top right",
                  annotation_font_color="#FF9800")
    fig.add_hline(y=80, line_dash="dot", line_color="#F44336", line_width=1,
                  annotation_text="80% danger", annotation_position="top right",
                  annotation_font_color="#F44336")

    # Phase boundaries
    for phase_name in ["Validation", "Backtest"]:
        ph = combined[combined["phase"] == phase_name]
        if not ph.empty:
            fig.add_vline(x=ph.index[0], line_dash="dash", line_color="gray", line_width=1.2)

    fig.update_layout(
        title="Drawdown Risk Budget (% of 10% MDD Limit)",
        xaxis_title="Date", yaxis_title="Budget Consumed (%)",
        yaxis=dict(range=[0, 105]),
        height=320, hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


def chart_trade_risk_impact(trades: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing ΔVaR (change in portfolio VaR %) for each trade.
    Red bars = risk increased; green bars = risk reduced.
    """
    if trades.empty:
        fig = go.Figure()
        fig.update_layout(title="Trade Risk Impact — No trades recorded", height=300)
        return fig

    colors = trades["delta_var"].apply(
        lambda x: "#F44336" if x > 0.001 else ("#4CAF50" if x < -0.001 else "#9E9E9E")
    )

    fig = go.Figure(go.Bar(
        x=trades["date"],
        y=trades["delta_var"],
        marker_color=colors,
        name="ΔVaR per Trade",
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "ΔVaR: %{y:.3f}%<br>"
            "<extra></extra>"
        ),
        text=trades["risk_action"],
        textposition="outside",
        textfont=dict(size=8),
    ))

    fig.add_hline(y=0, line_color="gray", line_width=0.8)

    # Phase labels on x-axis
    for phase_name, color in PHASE_COLORS.items():
        ph_trades = trades[trades["phase"] == phase_name]
        if not ph_trades.empty:
            mid_date = ph_trades["date"].iloc[len(ph_trades) // 2]
            fig.add_annotation(
                x=mid_date, y=trades["delta_var"].max() * 0.9,
                text=f"<b>{phase_name}</b>", showarrow=False,
                font=dict(size=9, color=color), bgcolor="white", opacity=0.7,
            )

    fig.update_layout(
        title="Trade Risk Impact — ΔVaR per Allocation Change",
        xaxis_title="Trade Date", yaxis_title="ΔVaR (% of Portfolio)",
        height=340, bargap=0.3,
        margin=dict(t=50, b=30),
    )
    return fig


def chart_risk_return_scatter(trades: pd.DataFrame, combined: pd.DataFrame) -> go.Figure:
    """
    Scatter: ΔVaR (risk taken) vs next-day portfolio return.
    Colour = regime; size = model confidence.
    """
    if trades.empty or len(trades) < 2:
        fig = go.Figure()
        fig.update_layout(title="Risk-Return Scatter — Insufficient data", height=340)
        return fig

    # Compute next-day portfolio return for each trade
    pv = combined["portfolio_value_norm"]
    next_returns = []
    for _, tr in trades.iterrows():
        dt = tr["date"]
        idx = combined.index.searchsorted(dt)
        if idx + 1 < len(combined):
            ret = (pv.iloc[idx + 1] - pv.iloc[idx]) / pv.iloc[idx] * 100
        else:
            ret = 0.0
        next_returns.append(ret)
    trades = trades.copy()
    trades["next_day_return"] = next_returns

    regime_color_map = {"Bull": "#4CAF50", "Bear": "#F44336", "Sideways": "#9E9E9E"}
    conf_vals = trades["model_confidence"].fillna(trades["model_confidence"].median())
    max_conf = conf_vals.max() if conf_vals.max() > 0 else 1.0
    sizes = (conf_vals / max_conf * 18 + 5).fillna(8)

    fig = go.Figure()
    for regime, rcolor in regime_color_map.items():
        mask = trades["regime"] == regime
        if not mask.any():
            continue
        sub = trades[mask]
        fig.add_trace(go.Scatter(
            x=sub["delta_var"],
            y=sub["next_day_return"],
            mode="markers",
            name=regime,
            marker=dict(
                color=rcolor, size=sizes[mask].tolist(),
                opacity=0.75, line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]|%Y-%m-%d}</b><br>"
                "ΔVaR: %{x:.3f}%<br>"
                "Next-day return: %{y:.2f}%<br>"
                "Phase: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=sub[["date", "phase"]].values,
        ))

    # Quadrant shading
    fig.add_hrect(y0=0, y1=trades["next_day_return"].max() + 1,
                  x0=trades["delta_var"].min() - 0.1, x1=0,
                  fillcolor="rgba(76,175,80,0.05)", line_width=0)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    # Quadrant labels
    fig.add_annotation(x=trades["delta_var"].min(), y=trades["next_day_return"].max(),
                       text="↑ High return / Low risk", showarrow=False,
                       font=dict(size=9, color="#2E7D32"), xanchor="left")
    fig.add_annotation(x=0.01, y=trades["next_day_return"].min(),
                       text="↓ Low return / High risk", showarrow=False,
                       font=dict(size=9, color="#B71C1C"), xanchor="left")

    fig.update_layout(
        title="Risk-Return per Trade (size = model confidence)",
        xaxis_title="ΔVaR % (risk added by trade)",
        yaxis_title="Next-day Portfolio Return (%)",
        height=360, hovermode="closest",
        margin=dict(t=50, b=30),
    )
    return fig


def chart_regime_confidence(combined: pd.DataFrame) -> go.Figure:
    """
    2-row subplot:
      Row 1: trend_score with coloured background bands (Bull/Bear/Sideways)
      Row 2: model confidence line
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["Market Regime (Trend Score)", "Model Confidence"],
        vertical_spacing=0.10,
    )

    # ── Row 1: trend_score + regime bands ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["trend_score"],
        name="Trend Score", line=dict(color="#1565C0", width=1.5),
        hovertemplate="%{y:.3f}",
    ), row=1, col=1)

    fig.add_hline(y=0.3, line_dash="dot", line_color="#4CAF50", line_width=0.8,
                  annotation_text="Bull threshold", row=1, col=1)
    fig.add_hline(y=-0.3, line_dash="dot", line_color="#F44336", line_width=0.8,
                  annotation_text="Bear threshold", row=1, col=1)
    fig.add_hrect(y0=0.3, y1=1.0, fillcolor="rgba(76,175,80,0.08)",
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=-1.0, y1=-0.3, fillcolor="rgba(244,67,54,0.08)",
                  line_width=0, row=1, col=1)

    # ── Row 2: model confidence ───────────────────────────────────────────────
    has_conf = combined["model_confidence"].notna().any()
    if has_conf:
        fig.add_trace(go.Scatter(
            x=combined.index, y=combined["model_confidence"],
            name="Confidence", line=dict(color="#9C27B0", width=1.3),
            fill="tozeroy", fillcolor="rgba(156,39,176,0.10)",
            hovertemplate="%{y:.3f}",
        ), row=2, col=1)
    else:
        fig.add_annotation(text="Q-value confidence not available",
                           xref="paper", yref="paper", x=0.5, y=0.15,
                           showarrow=False, font=dict(color="gray"))

    # Phase boundaries
    for phase_name in ["Validation", "Backtest"]:
        ph = combined[combined["phase"] == phase_name]
        if not ph.empty:
            for r in [1, 2]:
                fig.add_vline(x=ph.index[0], line_dash="dash",
                              line_color="gray", line_width=1.2, row=r, col=1)

    fig.update_yaxes(title_text="Trend Score", row=1, col=1)
    fig.update_yaxes(title_text="Q Spread", row=2, col=1)
    fig.update_layout(
        height=380, hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


# ─── Risk View tab renderer ────────────────────────────────────────────────────

def render_risk_view_tab(output_dir: str):
    """Render the 🛡️ Risk View tab — 6 charts built from risk_timeline.csv."""
    combined, trades, summary = RiskEngine.load(output_dir)

    if combined is None:
        st.info(
            "Risk data not yet available. "
            "Run the full pipeline (Steps 1–7) to generate the Risk View.",
            icon="ℹ️",
        )
        return

    # ── Summary metric cards ──────────────────────────────────────────────────
    st.subheader("Risk Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Starting VaR", "0.00%", "100% Cash")
    c2.metric(
        "Avg VaR (Invested)",
        f"{summary.get('avg_var_when_invested', 0):.2f}%",
        "when allocation > 0",
        delta_color="off",
    )
    c3.metric(
        "Peak VaR",
        f"{summary.get('peak_var_pct', 0):.2f}%",
        summary.get("peak_var_date", ""),
        delta_color="off",
    )
    c4.metric(
        "Max Budget Used",
        f"{summary.get('max_drawdown_budget_pct', 0):.1f}%",
        "of 10% MDD limit",
        delta_color="off",
    )
    c5.metric("Risk-Increasing Trades", str(summary.get("risk_increasing_trades", 0)))
    c6.metric("Risk-Reducing Trades", str(summary.get("risk_reducing_trades", 0)))

    # ── Per-phase breakdown table ─────────────────────────────────────────────
    per_phase = summary.get("per_phase", {})
    if per_phase:
        phase_rows = []
        for ph, vals in per_phase.items():
            phase_rows.append({
                "Phase": ph,
                "Days": vals.get("days", 0),
                "Trades": vals.get("trade_count", 0),
                "Avg Vol (%)": vals.get("avg_vol_pct", 0),
                "Avg VaR (%)": vals.get("avg_var_pct", 0),
                "Max DD (%)": vals.get("max_drawdown_pct", 0),
                "Max Budget (%)": vals.get("max_budget_used_pct", 0),
                "Sharpe": vals.get("sharpe_ratio", 0),
                "Return (%)": vals.get("total_return_pct", 0),
            })
        st.dataframe(
            pd.DataFrame(phase_rows),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ── Chart 1: Full Arc Risk Evolution ─────────────────────────────────────
    st.plotly_chart(chart_risk_evolution(combined), use_container_width=True)

    # ── Charts 2 & 3 side by side ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_drawdown_budget(combined), use_container_width=True)
    with col2:
        st.plotly_chart(chart_trade_risk_impact(trades), use_container_width=True)

    # ── Charts 4 & 5 side by side ─────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_risk_return_scatter(trades, combined), use_container_width=True)
    with col4:
        st.plotly_chart(chart_regime_confidence(combined), use_container_width=True)

    # ── Raw data expander ────────────────────────────────────────────────────
    with st.expander("Raw risk data (for export / report)"):
        st.caption("Risk Timeline (sampled)")
        st.dataframe(combined.iloc[::10], use_container_width=True)  # every 10th row
        st.caption("Trade Events")
        st.dataframe(trades, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇ risk_timeline.csv",
            data=combined.to_csv(),
            file_name="risk_timeline.csv",
            mime="text/csv",
            use_container_width=True,
        )
        col_b.download_button(
            "⬇ risk_trade_events.csv",
            data=trades.to_csv(index=False),
            file_name="risk_trade_events.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─── Training Progress tab renderer ───────────────────────────────────────────

def render_training_progress_tab(output_dir: str):
    """Render the 📈 Training Progress tab — round-by-round model improvement."""
    tracker = TrainingTracker(output_dir)
    history = tracker.load_history()

    if history.empty:
        st.info(
            "No training rounds recorded yet. "
            "Run the pipeline with 'Record this run to training history' checked.",
            icon="ℹ️",
        )
        return

    n_rounds = len(history)
    st.subheader(f"Training History — {n_rounds} Round{'s' if n_rounds > 1 else ''} Recorded")

    # ── Summary cards for latest round ───────────────────────────────────────
    latest = history.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Backtest Sharpe", f"{latest['backtest_sharpe']:.3f}",
              delta="✓ ≥1.0" if latest["backtest_sharpe"] >= 1.0 else "✗ <1.0",
              delta_color="normal" if latest["backtest_sharpe"] >= 1.0 else "inverse")
    c2.metric("Latest Backtest MDD", f"{latest['backtest_mdd_pct']:.1f}%",
              delta="✓ <10%" if abs(latest["backtest_mdd_pct"]) < 10 else "✗ Exceeded",
              delta_color="normal" if abs(latest["backtest_mdd_pct"]) < 10 else "inverse")
    c3.metric("Latest Backtest Return", f"{latest['backtest_return_pct']:.1f}%")
    c4.metric("Latest Val Sharpe", f"{latest['val_sharpe']:.3f}")

    st.divider()

    rounds = history["round"].tolist()
    x_labels = [f"R{r}" for r in rounds]

    # ── Chart A: Sharpe Ratio Progression ────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=x_labels, y=history["val_sharpe"].tolist(),
            name="Validation Sharpe", mode="lines+markers",
            line=dict(color="#6A1B9A", width=2),
            marker=dict(size=8),
        ))
        fig_sharpe.add_trace(go.Scatter(
            x=x_labels, y=history["backtest_sharpe"].tolist(),
            name="Backtest Sharpe", mode="lines+markers",
            line=dict(color="#2E7D32", width=2),
            marker=dict(size=8),
        ))
        fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="#F44336",
                             annotation_text="Target 1.0", annotation_position="top right")
        fig_sharpe.update_layout(
            title="Sharpe Ratio — Round by Round",
            xaxis_title="Round", yaxis_title="Sharpe Ratio",
            height=320, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Chart B: Max Drawdown Progression ────────────────────────────────────
    with col2:
        fig_mdd = go.Figure()
        fig_mdd.add_trace(go.Scatter(
            x=x_labels, y=history["val_mdd_pct"].abs().tolist(),
            name="Validation MDD", mode="lines+markers",
            line=dict(color="#6A1B9A", width=2, dash="dot"),
            marker=dict(size=8),
        ))
        fig_mdd.add_trace(go.Scatter(
            x=x_labels, y=history["backtest_mdd_pct"].abs().tolist(),
            name="Backtest MDD", mode="lines+markers",
            line=dict(color="#E65100", width=2),
            marker=dict(size=8),
        ))
        fig_mdd.add_hline(y=10, line_dash="dash", line_color="#F44336",
                          annotation_text="10% limit", annotation_position="top right")
        fig_mdd.update_layout(
            title="Max Drawdown — Round by Round (lower is better)",
            xaxis_title="Round", yaxis_title="Max Drawdown (%)",
            height=320, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_mdd, use_container_width=True)

    # ── Chart C: Backtest Return & CAGR ──────────────────────────────────────
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Bar(
        x=x_labels, y=history["backtest_return_pct"].tolist(),
        name="Total Return (%)", marker_color="#1565C0", opacity=0.8,
        text=[f"{v:.1f}%" for v in history["backtest_return_pct"]],
        textposition="outside",
    ))
    fig_ret.add_trace(go.Scatter(
        x=x_labels, y=history["backtest_cagr_pct"].tolist(),
        name="CAGR (%)", mode="lines+markers",
        line=dict(color="#FF9800", width=2),
        marker=dict(size=8),
    ))
    fig_ret.update_layout(
        title="Backtest Total Return & CAGR — Round by Round",
        xaxis_title="Round", yaxis_title="Return (%)",
        height=340, barmode="group",
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── Chart D: Constraint pass rate heatmap ─────────────────────────────────
    constraint_data = pd.DataFrame({
        "Round": x_labels,
        "Val Constraints": ["✓" if v else "✗" for v in history["val_constraints_met"]],
        "Backtest Constraints": ["✓" if v else "✗" for v in history["backtest_constraints_met"]],
        "Backtest Sharpe ≥1": ["✓" if v >= 1.0 else "✗" for v in history["backtest_sharpe"]],
        "MDD < 10%": ["✓" if abs(v) < 10 else "✗" for v in history["backtest_mdd_pct"]],
    })
    st.markdown("**Constraint Checklist per Round**")
    st.dataframe(constraint_data, use_container_width=True, hide_index=True)

    # ── Chart E: Training Loss & Reward ──────────────────────────────────────
    fig_tp = chart_training_progress(output_dir)
    if fig_tp is not None:
        st.plotly_chart(fig_tp, use_container_width=True)
    else:
        st.info(
            "Training Loss & Reward chart not available — model was loaded from cache. "
            "Enable *Force Retrain* and re-run Step 4 to generate this chart.",
            icon="💡",
        )

    # ── Full history table ────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Full Training History**")
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download training_history.csv",
        data=history.to_csv(index=False),
        file_name="training_history.csv",
        mime="text/csv",
    )


# ─── SHAP Explainability charts ───────────────────────────────────────────────

_ACTION_PALETTE = {"Cash (0%)": "#F44336", "Half (50%)": "#FF9800", "Full (100%)": "#4CAF50"}
_FEAT_COLOR = "#1565C0"


def chart_shap_global_importance(importance_df: pd.DataFrame) -> go.Figure:
    """
    Grouped horizontal bar chart: mean |SHAP| per feature per action.
    Answers: which features matter most, and for which action?
    """
    features = importance_df.index.tolist()
    fig = go.Figure()
    for action, color in _ACTION_PALETTE.items():
        if action in importance_df.columns:
            fig.add_trace(go.Bar(
                y=features,
                x=importance_df[action].tolist(),
                name=action,
                orientation="h",
                marker_color=color,
                opacity=0.85,
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            ))

    # Overall importance as scatter markers
    if "Overall" in importance_df.columns:
        fig.add_trace(go.Scatter(
            y=features,
            x=importance_df["Overall"].tolist(),
            mode="markers",
            name="Overall",
            marker=dict(color="black", size=9, symbol="diamond"),
            hovertemplate="<b>%{y}</b><br>Overall: %{x:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="Global Feature Importance — Mean |SHAP| per Feature × Action",
        xaxis_title="Mean |SHAP Value|",
        barmode="group",
        height=420,
        margin=dict(t=60, b=30, l=130),
        legend=dict(x=1.0, y=1),
    )
    return fig


def chart_shap_beeswarm(shap_values: np.ndarray, obs: np.ndarray, action_idx: int = 2) -> go.Figure:
    """
    Beeswarm-style scatter for one action (default: Full/100%).
    X = SHAP value  |  Y = feature  |  Colour = feature raw value (blue→red).
    Answers: how does each feature's actual value drive the Q-value for this action?
    """
    action_name = ACTION_NAMES[action_idx]
    shap_for_action = shap_values[:, :, action_idx]   # (N, 7)

    fig = go.Figure()
    for fi, fname in enumerate(FEATURE_NAMES):
        sv = shap_for_action[:, fi]
        fv = obs[:, fi]
        # Normalise feature value 0→1 for colour mapping
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        colors = [f"rgb({int(255*v)},{int(40*(1-v))},{int(200*(1-v))})" for v in fv_norm]

        fig.add_trace(go.Scatter(
            x=sv,
            y=[fname] * len(sv),
            mode="markers",
            name=fname,
            showlegend=False,
            marker=dict(color=colors, size=4, opacity=0.55),
            hovertemplate=(
                f"<b>{fname}</b><br>"
                "SHAP: %{x:.4f}<br>"
                "Feature value: %{customdata:.3f}<extra></extra>"
            ),
            customdata=fv,
        ))

    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")
    fig.update_layout(
        title=f"SHAP Beeswarm — Action: {action_name}  (colour = feature value: blue=low → red=high)",
        xaxis_title="SHAP Value (impact on Q-value)",
        height=430,
        margin=dict(t=60, b=30, l=130),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(FEATURE_NAMES))),
    )
    return fig


def chart_shap_per_action_comparison(importance_df: pd.DataFrame) -> go.Figure:
    """
    Radar / spider chart comparing how each feature ranks differently across actions.
    Answers: does MACD drive Cash decisions but RSI drives Full decisions?
    """
    features = importance_df.index.tolist()
    fig = go.Figure()
    for action, color in _ACTION_PALETTE.items():
        if action not in importance_df.columns:
            continue
        vals = importance_df[action].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=features + [features[0]],
            fill="toself",
            name=action,
            line_color=color,
            opacity=0.65,
        ))
    fig.update_layout(
        title="Feature Influence per Action — Radar Chart",
        polar=dict(radialaxis=dict(visible=True, title="Mean |SHAP|")),
        height=430,
        margin=dict(t=60, b=30),
    )
    return fig


def chart_shap_dependence(dep_data: dict, feature: str, action: str = "Full (100%)") -> go.Figure:
    """
    Dependence plot: feature raw value vs its SHAP value for one action.
    Answers: non-linear relationship between a feature and the model's preference.
    """
    action_key = {"Cash (0%)": "shap_cash", "Half (50%)": "shap_half", "Full (100%)": "shap_full"}
    ak = action_key.get(action, "shap_full")
    fd = dep_data.get(feature, {})
    x_vals = fd.get("values", np.array([]))
    y_vals = fd.get(ak, np.array([]))

    if len(x_vals) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {feature}", height=300)
        return fig

    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(
            color=y_vals,
            colorscale="RdYlGn",
            size=5, opacity=0.65,
            colorbar=dict(title="SHAP"),
        ),
        hovertemplate=f"<b>{feature}</b>: %{{x:.3f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Dependence Plot — {feature} → {action}",
        xaxis_title=f"{feature} (normalised value)",
        yaxis_title=f"SHAP Value for {action}",
        height=340,
        margin=dict(t=60, b=30),
    )
    return fig


def chart_shap_top_trades(top_trades_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar — per-feature SHAP contribution for the top N
    highest-magnitude trade decisions.
    Answers: what combination of signals led to the most decisive trades?
    """
    if top_trades_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No top trade data", height=300)
        return fig

    y_labels = [
        f"{str(row['date'])[:10]} → {row['action']}"
        for _, row in top_trades_df.iterrows()
    ]
    fig = go.Figure()
    palette = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#B71C1C", "#00695C", "#F57F17"]
    for fi, fname in enumerate(FEATURE_NAMES):
        if fname not in top_trades_df.columns:
            continue
        vals = top_trades_df[fname].tolist()
        fig.add_trace(go.Bar(
            y=y_labels,
            x=vals,
            name=fname,
            orientation="h",
            marker_color=palette[fi % len(palette)],
            opacity=0.85,
        ))

    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title="Top Decisive Trades — Per-Feature SHAP Contribution (chosen action)",
        xaxis_title="SHAP Value",
        barmode="relative",
        height=max(300, len(top_trades_df) * 42),
        margin=dict(t=60, b=30, l=210),
        legend=dict(x=1.0, y=1),
    )
    return fig


# ─── Explainability tab renderer ──────────────────────────────────────────────

def render_explainability_tab(output_dir: str):
    """Render the 🔍 Explainability tab — SHAP analysis of the DDQN agent."""
    shap_data = SHAPExplainer.load(output_dir)

    if shap_data is None:
        st.info(
            "SHAP data not yet available. "
            "Run the full pipeline (Steps 1–7) to generate explainability data.",
            icon="ℹ️",
        )
        return

    shap_vals = shap_data["shap_values"]      # (N, 7, 3)
    obs       = shap_data["observations"]     # (N, 7)
    phases    = shap_data["phases"]           # (N,)
    imp_df    = shap_data.get("importance")   # DataFrame (7, 4)

    n_samples = len(shap_vals)
    st.caption(
        f"SHAP values computed for **{n_samples}** observations "
        f"({', '.join(np.unique(phases).tolist())}) · "
        f"Background: {150} training samples · Method: DeepExplainer"
    )

    # ── Key insight cards ─────────────────────────────────────────────────────
    if imp_df is not None:
        top_feat = imp_df.index[0]
        top_cash = imp_df["Cash (0%)"].idxmax() if "Cash (0%)" in imp_df else "—"
        top_full = imp_df["Full (100%)"].idxmax() if "Full (100%)" in imp_df else "—"
        c1, c2, c3 = st.columns(3)
        c1.metric("Most Influential Feature (Overall)", top_feat)
        c2.metric("Top Driver for Cash (0%)", top_cash)
        c3.metric("Top Driver for Full (100%)", top_full)

    st.divider()

    # ── Chart 1: Global feature importance ───────────────────────────────────
    if imp_df is not None:
        st.plotly_chart(chart_shap_global_importance(imp_df), use_container_width=True)

    # ── Charts 2 & 3 side by side ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        action_choice = st.selectbox(
            "Beeswarm — select action to explain",
            ACTION_NAMES, index=2, key="beeswarm_action",
        )
        ai = ACTION_NAMES.index(action_choice)
        st.plotly_chart(
            chart_shap_beeswarm(shap_vals, obs, action_idx=ai),
            use_container_width=True,
        )
    with col2:
        if imp_df is not None:
            st.plotly_chart(chart_shap_per_action_comparison(imp_df), use_container_width=True)

    # ── Chart 4: Dependence plot (interactive feature selector) ──────────────
    st.subheader("Feature Dependence")
    col3, col4 = st.columns([1, 3])
    with col3:
        dep_feat = st.selectbox("Feature", FEATURE_NAMES, index=0, key="dep_feat")
        dep_action = st.selectbox("Action", ACTION_NAMES, index=2, key="dep_action")
    with col4:
        explainer_tmp = SHAPExplainer.__new__(SHAPExplainer)
        dep_data = explainer_tmp.dependence_data(shap_vals, obs)
        st.plotly_chart(
            chart_shap_dependence(dep_data, dep_feat, dep_action),
            use_container_width=True,
        )

    # ── Chart 5: Top decisive trades ─────────────────────────────────────────
    st.subheader("Most Decisive Trade Decisions")

    # We need allocations + dates aligned with explain_obs
    # These come from val + backtest episodes; reconstruct from risk_timeline
    combined_risk, _, _ = RiskEngine.load(output_dir)
    if combined_risk is not None:
        # Filter to Validation + Backtest phases (matching shap explain data)
        explain_phases = ["Validation", "Backtest"]
        sub = combined_risk[combined_risk["phase"].isin(explain_phases)].copy()
        allocs = sub["allocation"].tolist()
        dates_idx = sub.index

        explainer_tmp2 = SHAPExplainer.__new__(SHAPExplainer)
        n_align = min(len(shap_vals), len(allocs))
        top_trades = explainer_tmp2.top_trade_shap(
            shap_vals[:n_align], obs[:n_align],
            allocs[:n_align], dates_idx[:n_align + 1],
            n=10,
        )
        st.plotly_chart(chart_shap_top_trades(top_trades), use_container_width=True)

        with st.expander("Full top-trades table"):
            st.dataframe(top_trades, use_container_width=True, hide_index=True)

    # ── Per-phase importance comparison ──────────────────────────────────────
    phase_imp = shap_data.get("phase_importance", {})
    if len(phase_imp) > 1:
        st.subheader("Feature Importance — Validation vs Backtest")
        phase_cols = st.columns(len(phase_imp))
        for col, (ph, df_ph) in zip(phase_cols, phase_imp.items()):
            with col:
                st.markdown(f"**{ph}**")
                st.dataframe(
                    df_ph.style.background_gradient(cmap="Blues", axis=0),
                    use_container_width=True,
                )

    # ── Download ─────────────────────────────────────────────────────────────
    with st.expander("Download SHAP data"):
        if imp_df is not None:
            st.download_button(
                "⬇ shap_importance.csv",
                data=imp_df.to_csv(),
                file_name="shap_importance.csv",
                mime="text/csv",
            )


# ─── Dialog popups (one per chart — opened via "🔍 Expand" card button) ───────

@st.dialog("📈 Cumulative Returns", width="large")
def dlg_returns(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📉 Portfolio Drawdown", width="large")
def dlg_drawdown(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📊 Rolling Sharpe Ratio", width="large")
def dlg_sharpe_chart(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🎯 Allocation Decisions", width="large")
def dlg_allocations(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🧠 Q-Value Interpretability", width="large")
def dlg_qvals(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🛡️ Full Arc Risk Evolution", width="large")
def dlg_risk_evolution(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("💰 Drawdown Risk Budget", width="large")
def dlg_dd_budget(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("⚡ Trade Risk Impact", width="large")
def dlg_trade_risk(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🎯 Risk-Return Scatter", width="large")
def dlg_risk_return(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📡 Regime & Confidence", width="large")
def dlg_regime(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🔍 Global Feature Importance", width="large")
def dlg_shap_global(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🌊 SHAP Beeswarm", width="large")
def dlg_shap_beeswarm_dlg(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🕸️ Feature Radar", width="large")
def dlg_shap_radar(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📐 Dependence Plot", width="large")
def dlg_shap_dep(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("🏆 Top Decisive Trades", width="large")
def dlg_shap_trades(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📈 Sharpe Progress", width="large")
def dlg_train_sharpe(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📉 MDD Progress", width="large")
def dlg_train_mdd(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("💹 Return & CAGR Progress", width="large")
def dlg_train_return(fig): st.plotly_chart(fig, use_container_width=True)

@st.dialog("📉 Training Loss & Reward", width="large")
def dlg_train_progress(fig): st.plotly_chart(fig, use_container_width=True)


def chart_training_progress(output_dir: str):
    """
    Dual-axis Plotly chart: TD Loss (left Y) and Mean Episode Reward (right Y)
    over training timesteps. Reads training_progress.csv saved by TrainingProgressCallback.
    Returns None if the file does not exist (model loaded from cache, not retrained).
    """
    csv_path = os.path.join(output_dir, "training_progress.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path).dropna(subset=["timestep"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Loss trace (left Y)
    loss_df = df.dropna(subset=["loss"])
    if not loss_df.empty:
        fig.add_trace(
            go.Scatter(
                x=loss_df["timestep"], y=loss_df["loss"],
                name="TD Loss", mode="lines",
                line=dict(color="#E53935", width=1.5),
            ),
            secondary_y=False,
        )

    # Reward trace (right Y)
    rew_df = df.dropna(subset=["ep_rew_mean"])
    if not rew_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rew_df["timestep"], y=rew_df["ep_rew_mean"],
                name="Mean Episode Reward", mode="lines",
                line=dict(color="#1E88E5", width=1.5),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Training Progress — Loss & Reward over Timesteps",
        xaxis_title="Training Timestep",
        height=340, margin=dict(t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="TD Loss", secondary_y=False, title_font_color="#E53935")
    fig.update_yaxes(title_text="Mean Episode Reward", secondary_y=True, title_font_color="#1E88E5")
    return fig


# ─── UI helpers ───────────────────────────────────────────────────────────────

def chart_card(col, title: str, metrics: list, on_expand, btn_key: str, fig=None):
    """Card with mini-preview plot, summary metrics and an expand button."""
    import copy
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if fig is not None:
                preview = copy.deepcopy(fig)
                preview.update_layout(
                    height=140,
                    margin=dict(t=4, b=4, l=4, r=4),
                    showlegend=False,
                    title_text="",
                )
                preview.update_xaxes(title_text="", showticklabels=False)
                preview.update_yaxes(title_text="", showticklabels=False)
                st.plotly_chart(
                    preview, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"{btn_key}_preview",
                )
            for item in metrics:
                if len(item) == 3:
                    st.metric(item[0], item[1], item[2])
                else:
                    st.metric(item[0], item[1])
            if st.button("🔍 Expand", key=btn_key, use_container_width=True):
                on_expand()


def render_kpi_strip(perf: dict, risk_summary: dict = None, history=None):
    """8-metric global KPI strip rendered above all tabs."""
    tracker_rounds = len(history) if history is not None and not history.empty else 0
    avg_var = risk_summary.get("avg_var_when_invested", 0.0) if risk_summary else 0.0
    alpha = perf.get("portfolio_total_return_pct", 0) - perf.get("benchmark_total_return_pct", 0)
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Return", f"{perf.get('portfolio_total_return_pct', 0):.1f}%",
              delta=f"{alpha:+.1f}% vs BnH", delta_color="normal" if alpha >= 0 else "inverse")
    c2.metric("CAGR", f"{perf.get('portfolio_cagr_pct', 0):.1f}%",
              delta=f"BnH {perf.get('benchmark_cagr_pct', 0):.1f}%", delta_color="off")
    c3.metric("Sharpe", f"{perf.get('sharpe_ratio', 0):.3f}",
              delta="✓ ≥1.0" if perf.get("sharpe_constraint_met") else "✗ <1.0",
              delta_color="normal" if perf.get("sharpe_constraint_met") else "inverse")
    c4.metric("Max DD", f"{perf.get('max_drawdown_pct', 0):.1f}%",
              delta="✓ <10%" if perf.get("drawdown_constraint_met") else "✗ Exceeded",
              delta_color="normal" if perf.get("drawdown_constraint_met") else "inverse")
    c5.metric("Win Rate", f"{perf.get('win_rate_pct', 0):.1f}%",
              delta=f"{perf.get('total_days', 0)} days", delta_color="off")
    c6.metric("Avg VaR", f"{avg_var:.2f}%", delta="when invested", delta_color="off")
    c7.metric("Train Rounds", str(tracker_rounds))
    c8.metric("Constraints", "✓ All Met" if perf.get("constraints_met") else "✗ Failed",
              delta_color="off")


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

    # Load supporting data (used across multiple tabs)
    combined_risk, trades, risk_summary = RiskEngine.load(OUTPUT_DIR)
    tracker = TrainingTracker(OUTPUT_DIR)
    history = tracker.load_history()

    # ── Global KPI strip — always visible ─────────────────────────────────────
    render_kpi_strip(perf, risk_summary, history)
    st.divider()

    # ── 7-tab navigation ──────────────────────────────────────────────────────
    st.subheader("Results Dashboard")
    tab_dash, tab_steps, tab_perf, tab_risk, tab_explain, tab_train, tab_export = st.tabs([
        "🏠 Dashboard", "📋 Steps", "📈 Performance", "🛡️ Risk View",
        "🔍 Explainability", "📉 Training", "📥 Export",
    ])

    # ── 🏠 Dashboard ──────────────────────────────────────────────────────────
    with tab_dash:
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

    # ── 📋 Steps ──────────────────────────────────────────────────────────────
    with tab_steps:
        render_step_strip(step_results, OUTPUT_DIR)

    # ── 📈 Performance ────────────────────────────────────────────────────────
    with tab_perf:
        dates = backtest["dates"]
        port_vals = backtest["portfolio_values"]
        bench_vals = backtest["benchmark_values"]
        allocs = backtest["allocations"]
        closes = backtest["close_prices"]
        daily_rets = backtest["daily_returns"]
        q_vals = backtest.get("q_values_by_feature", {})

        # Precompute figures
        fig_ret   = chart_returns(port_vals, bench_vals, dates)
        fig_dd    = chart_drawdown(port_vals, dates)
        fig_rs    = chart_rolling_sharpe(daily_rets, dates)
        fig_alloc = chart_allocations(allocs, dates, closes)

        alpha = perf["portfolio_total_return_pct"] - perf["benchmark_total_return_pct"]
        port_arr  = np.asarray(port_vals)
        max_dd_val = float(((port_arr - np.maximum.accumulate(port_arr)) / np.maximum.accumulate(port_arr) * 100).min())

        # Row 1 — 4 cards
        c1, c2, c3, c4 = st.columns(4)
        chart_card(c1, "Cumulative Returns",
                   [("Total Return", f"{perf['portfolio_total_return_pct']:.1f}%", f"{alpha:+.1f}% vs BnH")],
                   lambda: dlg_returns(fig_ret), "btn_perf_ret", fig=fig_ret)
        chart_card(c2, "Portfolio Drawdown",
                   [("Max DD", f"{max_dd_val:.1f}%",
                     "✓ <10%" if perf["drawdown_constraint_met"] else "✗ Exceeded")],
                   lambda: dlg_drawdown(fig_dd), "btn_perf_dd", fig=fig_dd)
        chart_card(c3, "Rolling Sharpe",
                   [("Sharpe Ratio", f"{perf['sharpe_ratio']:.3f}",
                     "✓ ≥1.0" if perf["sharpe_constraint_met"] else "✗ <1.0")],
                   lambda: dlg_sharpe_chart(fig_rs), "btn_perf_sharpe", fig=fig_rs)
        chart_card(c4, "Allocation Decisions",
                   [("Win Rate", f"{perf['win_rate_pct']:.1f}%", f"{perf['total_days']} days")],
                   lambda: dlg_allocations(fig_alloc), "btn_perf_alloc", fig=fig_alloc)

        # Row 2 — Q-values + metrics summary
        if q_vals:
            fig_q = chart_q_values(q_vals)
            if fig_q:
                c5, c6 = st.columns(2)
                chart_card(c5, "Q-Value Interpretability",
                           [("Features", "RSI · MACD · BB · Trend"), ("Buckets", "Low / Mid / High")],
                           lambda: dlg_qvals(fig_q), "btn_perf_q", fig=fig_q)
                with c6:
                    with st.container(border=True):
                        st.markdown("**Backtest Summary**")
                        st.metric("CAGR", f"{perf['portfolio_cagr_pct']:.1f}%",
                                  delta=f"BnH {perf['benchmark_cagr_pct']:.1f}%", delta_color="off")
                        st.metric("Total Days", str(perf["total_days"]))

    # ── 🛡️ Risk View ──────────────────────────────────────────────────────────
    with tab_risk:
        if combined_risk is None:
            st.info("Risk data not yet available. Run the full pipeline (Steps 1–7).", icon="ℹ️")
        else:
            fig_re  = chart_risk_evolution(combined_risk)
            fig_ddb = chart_drawdown_budget(combined_risk)
            fig_tri = chart_trade_risk_impact(trades)
            fig_rrs = chart_risk_return_scatter(trades, combined_risk)
            fig_rc  = chart_regime_confidence(combined_risk)

            avg_var    = risk_summary.get("avg_var_when_invested", 0)
            peak_var   = risk_summary.get("peak_var_pct", 0)
            max_budget = risk_summary.get("max_drawdown_budget_pct", 0)
            ri_trades  = risk_summary.get("risk_increasing_trades", 0)
            rr_trades  = risk_summary.get("risk_reducing_trades", 0)

            # Row 1 — 4 cards
            c1, c2, c3, c4 = st.columns(4)
            chart_card(c1, "Risk Evolution",
                       [("Avg VaR", f"{avg_var:.2f}%"), ("Peak VaR", f"{peak_var:.2f}%")],
                       lambda: dlg_risk_evolution(fig_re), "btn_risk_evo", fig=fig_re)
            chart_card(c2, "DD Risk Budget",
                       [("Max Budget Used", f"{max_budget:.1f}%"), ("Limit", "10% MDD")],
                       lambda: dlg_dd_budget(fig_ddb), "btn_risk_ddb", fig=fig_ddb)
            chart_card(c3, "Trade Risk Impact",
                       [("Risk-Increasing", str(ri_trades)), ("Risk-Reducing", str(rr_trades))],
                       lambda: dlg_trade_risk(fig_tri), "btn_risk_tri", fig=fig_tri)
            chart_card(c4, "Regime & Confidence",
                       [("Phases", "Train · Val · Backtest"), ("Signal", "Trend + Q-Spread")],
                       lambda: dlg_regime(fig_rc), "btn_risk_rc", fig=fig_rc)

            # Row 2 — risk-return scatter + phase summary table
            c5, c6 = st.columns(2)
            chart_card(c5, "Risk-Return Scatter",
                       [("Regimes", "Bull / Bear / Sideways"), ("Size", "= Model Confidence")],
                       lambda: dlg_risk_return(fig_rrs), "btn_risk_rrs", fig=fig_rrs)
            with c6:
                with st.container(border=True):
                    st.markdown("**Phase Risk Summary**")
                    per_phase = risk_summary.get("per_phase", {})
                    if per_phase:
                        rows_ph = [
                            {"Phase": ph,
                             "Avg VaR%": f"{v.get('avg_var_pct', 0):.2f}",
                             "Max DD%":  f"{v.get('max_drawdown_pct', 0):.1f}",
                             "Trades":   v.get("trade_count", 0),
                             "Sharpe":   f"{v.get('sharpe_ratio', 0):.3f}",
                             "Return%":  f"{v.get('total_return_pct', 0):.1f}"}
                            for ph, v in per_phase.items()
                        ]
                        st.dataframe(pd.DataFrame(rows_ph), use_container_width=True, hide_index=True)

    # ── 🔍 Explainability ─────────────────────────────────────────────────────
    with tab_explain:
        shap_data = SHAPExplainer.load(OUTPUT_DIR)
        if shap_data is None:
            st.info("SHAP data not yet available. Run the full pipeline (Steps 1–7).", icon="ℹ️")
        else:
            shap_vals = shap_data["shap_values"]
            obs       = shap_data["observations"]
            imp_df    = shap_data.get("importance")

            top_feat  = imp_df.index[0] if imp_df is not None else "—"
            top_cash  = imp_df["Cash (0%)"].idxmax() if imp_df is not None and "Cash (0%)" in imp_df else "—"
            top_full  = imp_df["Full (100%)"].idxmax() if imp_df is not None and "Full (100%)" in imp_df else "—"

            fig_sg  = chart_shap_global_importance(imp_df) if imp_df is not None else None
            fig_sb  = chart_shap_beeswarm(shap_vals, obs, action_idx=2)
            fig_sr  = chart_shap_per_action_comparison(imp_df) if imp_df is not None else None
            _exp    = SHAPExplainer.__new__(SHAPExplainer)
            dep_data = _exp.dependence_data(shap_vals, obs)
            fig_sd  = chart_shap_dependence(dep_data, FEATURE_NAMES[0], ACTION_NAMES[2])

            fig_st = None
            if combined_risk is not None:
                sub = combined_risk[combined_risk["phase"].isin(["Validation", "Backtest"])].copy()
                alloc_list = sub["allocation"].tolist()
                n_align = min(len(shap_vals), len(alloc_list))
                top_trd = _exp.top_trade_shap(
                    shap_vals[:n_align], obs[:n_align],
                    alloc_list[:n_align], sub.index[:n_align + 1], n=10,
                )
                fig_st = chart_shap_top_trades(top_trd)

            # Row 1 — 4 cards
            c1, c2, c3, c4 = st.columns(4)
            if fig_sg:
                chart_card(c1, "Global Importance",
                           [("Top Feature", top_feat), ("Method", "Mean |SHAP|")],
                           lambda: dlg_shap_global(fig_sg), "btn_shap_global", fig=fig_sg)
            chart_card(c2, "SHAP Beeswarm",
                       [("Action", "Full (100%)"), ("Samples", str(len(shap_vals)))],
                       lambda: dlg_shap_beeswarm_dlg(fig_sb), "btn_shap_bee", fig=fig_sb)
            if fig_sr:
                chart_card(c3, "Feature Radar",
                           [("Top (Cash)", top_cash), ("Top (Full)", top_full)],
                           lambda: dlg_shap_radar(fig_sr), "btn_shap_radar", fig=fig_sr)
            chart_card(c4, "Dependence Plot",
                       [("Feature", FEATURE_NAMES[0]), ("Action", ACTION_NAMES[2])],
                       lambda: dlg_shap_dep(fig_sd), "btn_shap_dep", fig=fig_sd)

            # Row 2 — top trades + phase importance
            if fig_st is not None:
                c5, c6 = st.columns(2)
                chart_card(c5, "Top Decisive Trades",
                           [("Top N", "10 trades"), ("Ranking", "By |SHAP| magnitude")],
                           lambda: dlg_shap_trades(fig_st), "btn_shap_trades", fig=fig_st)
                with c6:
                    phase_imp = shap_data.get("phase_importance", {})
                    if phase_imp:
                        with st.container(border=True):
                            st.markdown("**Phase Importance (Overall)**")
                            for ph, df_ph in phase_imp.items():
                                st.caption(ph)
                                st.dataframe(df_ph[["Overall"]].head(4), use_container_width=True)

    # ── 📉 Training ───────────────────────────────────────────────────────────
    with tab_train:
        if history.empty:
            st.info(
                "No training rounds recorded yet. "
                "Run with 'Record this run to training history' checked.",
                icon="ℹ️",
            )
        else:
            latest   = history.iloc[-1]
            n_rounds = len(history)
            x_labels = [f"R{r}" for r in history["round"].tolist()]

            # Counter strip
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Rounds Recorded", str(n_rounds))
            tc2.metric("Latest Backtest Sharpe", f"{latest['backtest_sharpe']:.3f}",
                       delta="✓ ≥1.0" if latest["backtest_sharpe"] >= 1.0 else "✗ <1.0",
                       delta_color="normal" if latest["backtest_sharpe"] >= 1.0 else "inverse")
            tc3.metric("Latest Backtest MDD", f"{latest['backtest_mdd_pct']:.1f}%",
                       delta="✓ <10%" if abs(latest["backtest_mdd_pct"]) < 10 else "✗ Exceeded",
                       delta_color="normal" if abs(latest["backtest_mdd_pct"]) < 10 else "inverse")
            tc4.metric("Latest Return", f"{latest['backtest_return_pct']:.1f}%")

            # Build figures
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=x_labels, y=history["val_sharpe"].tolist(),
                                        name="Val Sharpe", mode="lines+markers",
                                        line=dict(color="#6A1B9A", width=2), marker=dict(size=8)))
            fig_ts.add_trace(go.Scatter(x=x_labels, y=history["backtest_sharpe"].tolist(),
                                        name="Backtest Sharpe", mode="lines+markers",
                                        line=dict(color="#2E7D32", width=2), marker=dict(size=8)))
            fig_ts.add_hline(y=1.0, line_dash="dash", line_color="#F44336",
                             annotation_text="Target 1.0", annotation_position="top right")
            fig_ts.update_layout(title="Sharpe Ratio — Round by Round",
                                 xaxis_title="Round", yaxis_title="Sharpe Ratio",
                                 height=320, margin=dict(t=50, b=30))

            fig_tm = go.Figure()
            fig_tm.add_trace(go.Scatter(x=x_labels, y=history["val_mdd_pct"].abs().tolist(),
                                        name="Val MDD", mode="lines+markers",
                                        line=dict(color="#6A1B9A", width=2, dash="dot"),
                                        marker=dict(size=8)))
            fig_tm.add_trace(go.Scatter(x=x_labels, y=history["backtest_mdd_pct"].abs().tolist(),
                                        name="Backtest MDD", mode="lines+markers",
                                        line=dict(color="#E65100", width=2), marker=dict(size=8)))
            fig_tm.add_hline(y=10, line_dash="dash", line_color="#F44336",
                             annotation_text="10% limit", annotation_position="top right")
            fig_tm.update_layout(title="Max Drawdown — Round by Round",
                                 xaxis_title="Round", yaxis_title="Max DD (%)",
                                 height=320, margin=dict(t=50, b=30))

            fig_tr = go.Figure()
            fig_tr.add_trace(go.Bar(x=x_labels, y=history["backtest_return_pct"].tolist(),
                                    name="Total Return (%)", marker_color="#1565C0", opacity=0.8,
                                    text=[f"{v:.1f}%" for v in history["backtest_return_pct"]],
                                    textposition="outside"))
            fig_tr.add_trace(go.Scatter(x=x_labels, y=history["backtest_cagr_pct"].tolist(),
                                        name="CAGR (%)", mode="lines+markers",
                                        line=dict(color="#FF9800", width=2), marker=dict(size=8)))
            fig_tr.update_layout(title="Backtest Return & CAGR — Round by Round",
                                 xaxis_title="Round", yaxis_title="Return (%)",
                                 height=320, barmode="group", margin=dict(t=50, b=30))

            # Row 1 — 4 cards
            c1, c2, c3, c4 = st.columns(4)
            chart_card(c1, "Sharpe Progress",
                       [("Latest Val", f"{latest['val_sharpe']:.3f}"),
                        ("Latest Backtest", f"{latest['backtest_sharpe']:.3f}")],
                       lambda: dlg_train_sharpe(fig_ts), "btn_train_sharpe", fig=fig_ts)
            chart_card(c2, "MDD Progress",
                       [("Latest Val MDD", f"{latest['val_mdd_pct']:.1f}%"),
                        ("Latest Backtest MDD", f"{latest['backtest_mdd_pct']:.1f}%")],
                       lambda: dlg_train_mdd(fig_tm), "btn_train_mdd", fig=fig_tm)
            chart_card(c3, "Return & CAGR",
                       [("Latest Return", f"{latest['backtest_return_pct']:.1f}%"),
                        ("Latest CAGR", f"{latest['backtest_cagr_pct']:.1f}%")],
                       lambda: dlg_train_return(fig_tr), "btn_train_ret", fig=fig_tr)
            with c4:
                with st.container(border=True):
                    st.markdown("**Constraint Checklist**")
                    val_ok = latest.get("val_constraints_met", False)
                    bt_ok  = latest.get("backtest_constraints_met", False)
                    st.markdown(f"Validation: {'✓' if val_ok else '✗'}")
                    st.markdown(f"Backtest:   {'✓' if bt_ok else '✗'}")
                    st.caption(f"{n_rounds} round{'s' if n_rounds > 1 else ''} recorded")

            # Row 2 — Training Loss & Reward (full width, only if retrained)
            fig_tp = chart_training_progress(OUTPUT_DIR)
            if fig_tp is not None:
                cp1, cp2 = st.columns([3, 1])
                chart_card(cp1, "Training Loss & Reward",
                           [("X-axis", "Timestep"), ("Loss", "TD (red) · Reward (blue)")],
                           lambda: dlg_train_progress(fig_tp), "btn_train_prog", fig=fig_tp)
                with cp2:
                    with st.container(border=True):
                        st.markdown("**About this chart**")
                        st.caption(
                            "TD Loss (left axis): lower = more stable Q-value estimates. "
                            "Mean Episode Reward (right axis): higher = better strategy quality. "
                            "Only available when model is retrained (force_retrain=True)."
                        )
            else:
                st.caption("💡 Training Loss & Reward chart available after retraining with *Force Retrain* enabled.")

            # Full history table (collapsed)
            with st.expander("Full training history"):
                st.dataframe(history, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ training_history.csv",
                    data=history.to_csv(index=False),
                    file_name="training_history.csv",
                    mime="text/csv",
                )

    # ── 📥 Export ─────────────────────────────────────────────────────────────
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

        if combined_risk is not None:
            col_c, col_d = st.columns(2)
            col_c.download_button(
                "⬇ risk_timeline.csv",
                data=combined_risk.to_csv(),
                file_name="risk_timeline.csv",
                mime="text/csv",
                use_container_width=True,
            )
            col_d.download_button(
                "⬇ risk_trade_events.csv",
                data=trades.to_csv(index=False),
                file_name="risk_trade_events.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
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

    with st.status("Step 7 — Generating results, risk arc & plots...", expanded=False) as s:
        r = orch.step7_get_results(
            record_to_history=config.get("record_run", True),
            notes=config.get("run_notes", ""),
        )
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
