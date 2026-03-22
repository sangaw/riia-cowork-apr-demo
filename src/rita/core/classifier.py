"""
RITA Core — Intent Classifier

Classifies free-text user investment queries into 20 fixed scenarios using
all-MiniLM-L6-v2 cosine similarity over seed phrases.

No Claude API call at runtime.  All responses are derived from OHLCV data via
deterministic core handlers.  Episode-dependent intents read persisted CSVs
from rita_output/ so they work even without a live workflow session.

Usage
-----
    from rita.core.classifier import classify, dispatch

    result = classify("can I invest in Nifty now?")
    if not result.low_confidence:
        reply = dispatch(result, df)
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Singleton model (copied from arch-reference rag/retriever.py) ────────────

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_seed_embeddings: Optional[np.ndarray] = None   # (N, 384) float32, normalised
_seed_to_intent: list[int] = []                  # seed row → intent index

CONFIDENCE_THRESHOLD = 0.42   # cosine sim; below this = unreliable match


def _get_model():
    """Lazy-load all-MiniLM-L6-v2 once; reuse on every subsequent call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ─── Intent definition ────────────────────────────────────────────────────────

@dataclass
class Intent:
    name: str
    seeds: list[str]
    handler: str    # market_sentiment | strategy_recommendation | return_estimates
                    # stress_scenarios | performance_feedback | portfolio_comparison
    params: dict = field(default_factory=dict)
    template: str = ""   # {key} placeholders filled by dispatch()


INTENTS: list[Intent] = [

    # ── Market Sentiment (4 intents) ─────────────────────────────────────────

    Intent(
        name="market_sentiment",
        seeds=[
            "what is the current market sentiment",
            "is the market bullish or bearish",
            "how is the market feeling today",
            "overall market mood nifty",
            "what is investor sentiment right now",
            "bullish or bearish signal today",
            "what does the market analysis say",
        ],
        handler="market_sentiment",
        template=(
            "Market sentiment: **{overall_sentiment}** (score {total_score:+d}/6). "
            "{signal_summary}."
        ),
    ),

    Intent(
        name="trend_direction",
        seeds=[
            "what direction is nifty trending",
            "is it an uptrend or downtrend",
            "which way is nifty heading",
            "current price trend for nifty",
            "is the market in an uptrend",
            "EMA trend direction today",
            "short term vs long term moving average nifty",
        ],
        handler="market_sentiment",
        template=(
            "Nifty is in a **{trend}** (trend score {trend_score:+.3f}). "
            "EMA-50: ₹{ema_50:,.0f} | EMA-200: ₹{ema_200:,.0f}."
        ),
    ),

    Intent(
        name="rsi_reading",
        seeds=[
            "what is the RSI reading today",
            "is the market overbought or oversold",
            "RSI signal for nifty right now",
            "relative strength index level nifty",
            "momentum indicator reading nifty",
            "RSI based buy or sell signal nifty",
        ],
        handler="market_sentiment",
        template=(
            "RSI-14: **{rsi_14:.1f}** ({rsi_signal}). "
            "Close: ₹{close:,.0f} as of {date}."
        ),
    ),

    Intent(
        name="volatility_check",
        seeds=[
            "how volatile is nifty right now",
            "current market volatility level",
            "what is nifty volatility today",
            "ATR reading for nifty",
            "how much is nifty moving each day",
            "daily price swings in nifty",
            "is volatility high or low today",
        ],
        handler="market_sentiment",
        template=(
            "ATR-14: ₹{atr_14:,.0f} (at {atr_pct:.0f}th percentile of history). "
            "Volatility: **{volatility_label}**. Trend: {trend}."
        ),
    ),

    # ── Strategy / Allocation (4 intents) ────────────────────────────────────

    Intent(
        name="invest_now",
        seeds=[
            "can I invest in nifty now",
            "should I enter the market today",
            "is now a good time to buy nifty",
            "should I invest right now",
            "is this a good entry point for nifty",
            "right time to put money in nifty",
            "what is the risk of investing in nifty today",
            "should I buy nifty or wait",
        ],
        handler="strategy_recommendation",
        template=(
            "Recommended action: **{recommendation}** ({allocation_pct}% invested). "
            "{rationale}"
        ),
    ),

    Intent(
        name="allocation_level",
        seeds=[
            "what allocation should I have in nifty",
            "how much of my portfolio should be in equities",
            "optimal portfolio allocation for nifty",
            "what percentage of capital to invest in nifty",
            "nifty allocation recommendation today",
            "how much exposure to nifty should I have",
        ],
        handler="strategy_recommendation",
        template=(
            "Suggested allocation: **{allocation_pct}%** ({recommendation}). "
            "Constraint: {primary_constraint}. {upgrade_trigger}"
        ),
    ),

    Intent(
        name="conservative_strategy",
        seeds=[
            "safe investment approach for nifty",
            "conservative allocation for nifty",
            "low risk nifty strategy",
            "I want to protect my capital in nifty",
            "minimum risk nifty investment",
            "capital protection strategy nifty",
            "risk-averse investor nifty approach",
        ],
        handler="strategy_recommendation",
        params={"hint": "conservative"},
        template=(
            "Conservative guidance: signals recommend **{recommendation}** ({allocation_pct}%). "
            "{rationale}"
        ),
    ),

    Intent(
        name="aggressive_strategy",
        seeds=[
            "aggressive nifty investment strategy",
            "maximum allocation to nifty",
            "high risk high reward in nifty",
            "I can tolerate high risk in nifty",
            "full equity allocation nifty",
            "go all in on nifty",
            "aggressive investor nifty recommendation",
        ],
        handler="strategy_recommendation",
        params={"hint": "aggressive"},
        template=(
            "Aggressive guidance: model recommends **{recommendation}** ({allocation_pct}%). "
            "Ensure drawdown tolerance. {downgrade_trigger}"
        ),
    ),

    # ── Return Estimates (6 intents) ─────────────────────────────────────────

    Intent(
        name="return_1m",
        seeds=[
            "what returns can I expect in 1 month from nifty",
            "nifty outlook for next month",
            "monthly return estimate nifty",
            "nifty return next 30 days",
            "1 month investment return nifty",
            "short term return 1 month nifty",
        ],
        handler="return_estimates",
        params={"period_days": 21, "label": "1 month"},
        template=(
            "Historical {label} returns — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    Intent(
        name="return_3m",
        seeds=[
            "3 month nifty return estimate",
            "quarterly outlook for nifty",
            "nifty returns over next 3 months",
            "short to medium term return nifty",
            "3 month investment horizon nifty",
            "quarterly return nifty forecast",
        ],
        handler="return_estimates",
        params={"period_days": 91, "label": "3 months"},
        template=(
            "Historical {label} returns — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    Intent(
        name="return_6m",
        seeds=[
            "6 month nifty return estimate",
            "half year nifty outlook",
            "nifty returns over next 6 months",
            "6 month investment estimate nifty",
            "semi-annual return expectation nifty",
            "nifty performance over half a year",
        ],
        handler="return_estimates",
        params={"period_days": 182, "label": "6 months"},
        template=(
            "Historical {label} returns — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    Intent(
        name="return_1y",
        seeds=[
            "annual return from nifty",
            "1 year investment return nifty",
            "yearly nifty return estimate",
            "how much will nifty give in 1 year",
            "one year outlook nifty",
            "expected annual return nifty",
            "nifty 12 month return",
        ],
        handler="return_estimates",
        params={"period_days": 365, "label": "1 year"},
        template=(
            "Historical {label} CAGR — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    Intent(
        name="return_3y",
        seeds=[
            "3 year nifty return estimate",
            "medium term return nifty 3 years",
            "how much does nifty grow in 3 years",
            "3 year CAGR nifty estimate",
            "compound return over 3 years nifty",
            "nifty wealth creation 3 years",
        ],
        handler="return_estimates",
        params={"period_days": 1095, "label": "3 years"},
        template=(
            "Historical {label} CAGR — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    Intent(
        name="return_5y",
        seeds=[
            "5 year nifty return estimate",
            "long term investment return nifty",
            "how much will nifty give in 5 years",
            "5 year wealth creation nifty",
            "long horizon return estimate nifty",
            "nifty CAGR over 5 years",
        ],
        handler="return_estimates",
        params={"period_days": 1825, "label": "5 years"},
        template=(
            "Historical {label} CAGR — median **{p50:.1f}%** "
            "(best 25%: +{p75:.1f}%, worst 25%: {p25:.1f}%, win rate: {win_rate:.0f}%)."
        ),
    ),

    # ── Stress Scenarios (4 intents) ─────────────────────────────────────────

    Intent(
        name="stress_crash_10",
        seeds=[
            "what if nifty falls 10 percent",
            "10 percent market correction impact on portfolio",
            "nifty drops 10 percent what happens",
            "mild correction scenario 10 percent",
            "impact of 10 percent nifty decline",
            "10 percent downside scenario nifty",
        ],
        handler="stress_scenarios",
        params={"move": -10},
        template=(
            "If Nifty falls 10%: RITA ({rita_pct}% invested) → **{rita_impact:.1f}%** "
            "vs Aggressive −10.0%, Moderate −6.0%, Conservative −3.0%. "
            "Drawdown breach (>10%): {breach_note}."
        ),
    ),

    Intent(
        name="stress_crash_20",
        seeds=[
            "nifty crash 20 percent scenario",
            "severe bear market impact on portfolio",
            "what if market crashes 20 percent",
            "deep correction 20 percent fall nifty",
            "worst case bear market scenario nifty",
            "nifty 20 percent downside risk",
        ],
        handler="stress_scenarios",
        params={"move": -20},
        template=(
            "If Nifty falls 20%: RITA ({rita_pct}% invested) → **{rita_impact:.1f}%** "
            "vs Aggressive −20.0%, Moderate −12.0%, Conservative −6.0%. "
            "Drawdown breach: {breach_note}."
        ),
    ),

    Intent(
        name="stress_rally_10",
        seeds=[
            "what if nifty rises 10 percent",
            "10 percent rally scenario nifty",
            "bull market 10 percent gain portfolio",
            "nifty up 10 percent portfolio effect",
            "strong nifty rally scenario",
            "10 percent upside scenario nifty",
        ],
        handler="stress_scenarios",
        params={"move": 10},
        template=(
            "If Nifty rallies 10%: RITA ({rita_pct}% invested) → **+{rita_impact:.1f}%** "
            "vs Aggressive +10.0%, Moderate +6.0%, Conservative +3.0%."
        ),
    ),

    Intent(
        name="stress_flat",
        seeds=[
            "sideways market scenario nifty",
            "nifty stays flat what happens to portfolio",
            "no market movement scenario",
            "range-bound nifty impact",
            "flat nifty scenario portfolio effect",
            "nifty going nowhere impact on investment",
        ],
        handler="stress_scenarios",
        params={"move": 0},
        template=(
            "If Nifty stays flat (0%): all profiles show ~0% portfolio change. "
            "RITA current stance: {recommendation} ({rita_pct}% invested)."
        ),
    ),

    # ── Performance & Portfolio (2 intents) ──────────────────────────────────

    Intent(
        name="backtest_performance",
        seeds=[
            "how has RITA model performed historically",
            "show me past performance results RITA",
            "backtest results summary RITA",
            "RITA historical performance analysis",
            "what is the RITA model track record",
            "has RITA beaten the benchmark",
            "RITA trading model performance metrics",
        ],
        handler="performance_feedback",
        template=(
            "RITA backtest: Sharpe **{sharpe:.3f}**, MDD **{mdd_pct:.2f}%**, "
            "Return **{total_return_pct:.1f}%** vs benchmark **{benchmark_pct:.1f}%**. "
            "{summary_note}"
        ),
    ),

    Intent(
        name="portfolio_compare",
        seeds=[
            "compare different portfolio allocation strategies nifty",
            "how do conservative moderate aggressive portfolios compare",
            "portfolio scenario comparison nifty",
            "which allocation strategy is best for nifty",
            "show conservative vs aggressive nifty comparison",
            "compare RITA to buy and hold nifty",
        ],
        handler="portfolio_comparison",
        template=(
            "Portfolio comparison — Conservative: {conservative_ret:.1f}%, "
            "Moderate: {moderate_ret:.1f}%, Aggressive: {aggressive_ret:.1f}%, "
            "RITA: **{rita_ret:.1f}%**."
        ),
    ),
]


# ─── Seed index ───────────────────────────────────────────────────────────────

def _build_seed_index() -> None:
    """Encode all seed phrases once; store normalised embeddings and intent map."""
    global _seed_embeddings, _seed_to_intent
    model = _get_model()
    seeds, mapping = [], []
    for i, intent in enumerate(INTENTS):
        for s in intent.seeds:
            seeds.append(s)
            mapping.append(i)
    _seed_embeddings = model.encode(
        seeds, normalize_embeddings=True, show_progress_bar=False
    )
    _seed_to_intent = mapping


# ─── Classification ───────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: Intent
    confidence: float      # cosine similarity of best-matching seed
    low_confidence: bool   # True if confidence < CONFIDENCE_THRESHOLD


def classify(query: str, threshold: float = CONFIDENCE_THRESHOLD) -> IntentResult:
    """
    Classify a free-text investment query into one of the 20 RITA intents.

    Args:
        query     : Natural-language investment question.
        threshold : Minimum cosine similarity for a reliable match.

    Returns:
        IntentResult with matched intent, similarity score, and low_confidence flag.
        When low_confidence is True consider returning a fallback response.
    """
    global _seed_embeddings, _seed_to_intent
    if _seed_embeddings is None:
        _build_seed_index()

    model = _get_model()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = (_seed_embeddings @ q_emb).astype(float)   # (N,) cosine similarities
    best = int(np.argmax(sims))
    confidence = float(sims[best])

    return IntentResult(
        intent=INTENTS[_seed_to_intent[best]],
        confidence=confidence,
        low_confidence=confidence < threshold,
    )


# ─── Dispatch ─────────────────────────────────────────────────────────────────

_NO_PERF_MSG = (
    "No backtest data found. Run the RITA pipeline (step 6) first "
    "to generate performance data."
)


def _load_perf_summary(output_dir: str) -> Optional[dict]:
    """Read rita_output/performance_summary.csv into a flat {metric: float} dict."""
    path = os.path.join(output_dir, "performance_summary.csv")
    if not os.path.exists(path):
        return None
    perf: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                perf[row["metric"]] = float(row["value"])
            except (ValueError, KeyError):
                perf[row.get("metric", "")] = row.get("value", "")
    return perf or None


def dispatch(
    result: IntentResult,
    df,                                   # pd.DataFrame with indicators applied
    *,
    portfolio_inr: float = 1_000_000,
    output_dir: str = "./rita_output",
) -> str:
    """
    Run the deterministic OHLCV handler for the classified intent and return
    the filled template string.

    Args:
        result        : Output of classify().
        df            : Nifty DataFrame post calculate_indicators().
        portfolio_inr : Starting capital for stress / portfolio scenarios (default ₹10L).
        output_dir    : Path to rita_output/ — used to read persisted CSVs for
                        performance_feedback and portfolio_comparison intents.

    Returns:
        Human-readable response string, or a graceful fallback message.
    """
    from rita.core.technical_analyzer import get_market_summary, get_sentiment_score
    from rita.core.strategy_engine import get_allocation_recommendation
    from rita.core.data_loader import get_period_return_estimates
    from rita.core.performance import simulate_stress_scenarios, build_portfolio_comparison

    intent = result.intent
    params = intent.params
    h = intent.handler

    # ── Market sentiment / technical ──────────────────────────────────────────
    if h == "market_sentiment":
        summary = get_market_summary(df)
        scored = get_sentiment_score(summary)
        ap = summary["atr_percentile"]
        ctx = {
            **summary,
            **scored,
            "atr_pct": round(ap * 100),
            "volatility_label": (
                "high"   if ap > 0.75 else
                "low"    if ap < 0.35 else
                "normal"
            ),
        }
        return intent.template.format(**ctx)

    # ── Strategy / allocation ─────────────────────────────────────────────────
    elif h == "strategy_recommendation":
        summary = get_market_summary(df)
        scored = get_sentiment_score(summary)
        rec = get_allocation_recommendation(summary, scored)
        return intent.template.format(**rec)

    # ── Return estimates ──────────────────────────────────────────────────────
    elif h == "return_estimates":
        est = get_period_return_estimates(df, params["period_days"])
        s = est["scenarios"]
        ctx = {
            "label":    params["label"],
            "p50":      s["median"]["annualized_pct"],
            "p25":      s["cautious"]["annualized_pct"],
            "p75":      s["optimistic"]["annualized_pct"],
            "win_rate": est["win_rate_pct"],
        }
        return intent.template.format(**ctx)

    # ── Stress scenarios ──────────────────────────────────────────────────────
    elif h == "stress_scenarios":
        move: int = params["move"]
        summary = get_market_summary(df)
        scored = get_sentiment_score(summary)
        rec = get_allocation_recommendation(summary, scored)
        rita_alloc = rec["allocation_pct"]

        stress = simulate_stress_scenarios(portfolio_inr, [move], rita_alloc)
        move_key = f"{move:+d}%"
        rita = stress["scenarios"][move_key]["profiles"]["rita_current"]

        ctx = {
            "rita_pct":       int(rita_alloc),
            "rita_impact":    rita["portfolio_impact_pct"],
            "breach_note":    "YES" if rita["breaches_10pct_dd"] else "No",
            "recommendation": rec["recommendation"],
        }
        return intent.template.format(**ctx)

    # ── Performance feedback — reads persisted CSV ────────────────────────────
    elif h == "performance_feedback":
        perf = _load_perf_summary(output_dir)
        if perf is None:
            return _NO_PERF_MSG
        sharpe = perf.get("sharpe_ratio", 0.0)
        ctx = {
            "sharpe":           sharpe,
            "mdd_pct":          perf.get("max_drawdown_pct", 0.0),
            "total_return_pct": perf.get("portfolio_total_return_pct", 0.0),
            "benchmark_pct":    perf.get("benchmark_total_return_pct", 0.0),
            "summary_note": (
                "Sharpe > 1.0 target met ✓"
                if sharpe > 1.0 else
                "Sharpe below 1.0 — consider retraining."
            ),
        }
        return intent.template.format(**ctx)

    # ── Portfolio comparison — reads backtest_daily.csv ───────────────────────
    elif h == "portfolio_comparison":
        import pandas as pd
        daily_path = os.path.join(output_dir, "backtest_daily.csv")
        if not os.path.exists(daily_path):
            return _NO_PERF_MSG
        backtest_df = pd.read_csv(daily_path)
        comparison = build_portfolio_comparison(backtest_df, portfolio_inr)
        p = comparison["profiles"]
        ctx = {
            "conservative_ret": p["conservative"]["return_pct"],
            "moderate_ret":     p["moderate"]["return_pct"],
            "aggressive_ret":   p["aggressive"]["return_pct"],
            "rita_ret":         p["rita_model"]["return_pct"],
        }
        return intent.template.format(**ctx)

    return f"[No handler registered for intent: {intent.name}]"
