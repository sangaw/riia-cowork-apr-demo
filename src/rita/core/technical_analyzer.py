"""
RITA Core — Technical Analyzer
Calculates technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA)
using the `ta` library on the Nifty 50 OHLCV DataFrame.
"""

import numpy as np
import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to the OHLCV DataFrame.

    Adds:
        rsi_14        — RSI (14-period)
        macd          — MACD line
        macd_signal   — MACD signal line
        macd_hist     — MACD histogram
        bb_upper      — Bollinger upper band
        bb_mid        — Bollinger middle band
        bb_lower      — Bollinger lower band
        bb_pct_b      — %B position (0=lower, 1=upper)
        atr_14        — ATR (14-period)
        ema_50        — 50-day EMA
        ema_200       — 200-day EMA
        trend_score   — normalized slope of ema_50 (-1 to +1)
        ema_ratio     — ema_26 / ema_50, clipped to [0.5, 1.5] (Feature 9 — regime signal)
        daily_return  — daily close-to-close return
    """
    out = df.copy()

    # RSI
    out["rsi_14"] = ta.momentum.RSIIndicator(close=out["Close"], window=14).rsi()

    # MACD
    macd_ind = ta.trend.MACD(close=out["Close"], window_fast=12, window_slow=26, window_sign=9)
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_ind.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=out["Close"], window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_pct_b"] = bb.bollinger_pband()

    # ATR
    out["atr_14"] = ta.volatility.AverageTrueRange(
        high=out["High"], low=out["Low"], close=out["Close"], window=14
    ).average_true_range()

    # EMAs
    out["ema_5"]   = ta.trend.EMAIndicator(close=out["Close"], window=5).ema_indicator()
    out["ema_13"]  = ta.trend.EMAIndicator(close=out["Close"], window=13).ema_indicator()
    out["ema_26"]  = ta.trend.EMAIndicator(close=out["Close"], window=26).ema_indicator()
    out["ema_50"]  = ta.trend.EMAIndicator(close=out["Close"], window=50).ema_indicator()
    out["ema_200"] = ta.trend.EMAIndicator(close=out["Close"], window=200).ema_indicator()

    # Trend score: normalized slope of ema_50 over last 20 days
    def _rolling_trend_score(series: pd.Series, window: int = 20) -> pd.Series:
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                segment = series.iloc[i - window + 1: i + 1].values
                x = np.arange(window, dtype=float)
                slope = np.polyfit(x, segment, 1)[0]
                # Normalize by price level
                slopes.append(slope / series.iloc[i])
        return pd.Series(slopes, index=series.index)

    raw_trend = _rolling_trend_score(out["ema_50"].dropna())
    out["trend_score"] = raw_trend.reindex(out.index)
    # Clip to [-0.01, +0.01] then scale to [-1, +1]
    out["trend_score"] = out["trend_score"].clip(-0.01, 0.01) / 0.01

    # EMA ratio — medium-term regime context (Feature 9)
    # EMA-26 already computed above; ratio > 1 = uptrend, < 1 = downtrend
    # Normalized: clip((ratio - 1.0) * 20, -3, 3)
    out["ema_ratio"] = (out["ema_26"] / out["ema_50"]).clip(0.5, 1.5)

    # Daily return
    out["daily_return"] = out["Close"].pct_change()

    return out


def detect_regime(df: pd.DataFrame, consecutive_days: int = 3) -> dict:
    """
    Detect the current market regime using the EMA-26/EMA-50 ratio.

    Rule:
        ema_26 / ema_50 < 0.99 for `consecutive_days` or more  → BEAR
        otherwise                                                → BULL

    Args:
        df: DataFrame already processed by calculate_indicators()
        consecutive_days: number of days below 0.99 required to call BEAR (default 3)

    Returns:
        {
            "regime":               "BULL" | "BEAR",
            "model":                "bull" | "bear",
            "ema_ratio":            float,          # latest EMA-26/EMA-50
            "consecutive_bear_days": int,           # days below 0.99 at tail
        }
    """
    clean = df.dropna(subset=["ema_26", "ema_50"])
    if clean.empty:
        return {"regime": "BULL", "model": "bull", "ema_ratio": 1.0, "consecutive_bear_days": 0}

    ratio_series = clean["ema_26"] / clean["ema_50"]
    latest_ratio = float(ratio_series.iloc[-1])

    # Count consecutive bear days at the tail
    bear_mask = ratio_series < 0.99
    count = 0
    for val in reversed(bear_mask.values):
        if val:
            count += 1
        else:
            break

    regime = "BEAR" if count >= consecutive_days else "BULL"
    return {
        "regime": regime,
        "model": "bear" if regime == "BEAR" else "bull",
        "ema_ratio": round(latest_ratio, 5),
        "consecutive_bear_days": count,
    }


def get_market_summary(df: pd.DataFrame) -> dict:
    """
    Return a human-readable summary of the latest market indicators.
    Expects a DataFrame already processed by calculate_indicators().
    """
    latest = df.dropna(subset=["rsi_14", "macd", "ema_50", "ema_200"]).iloc[-1]

    # Trend classification
    if latest["ema_50"] > latest["ema_200"] and latest["trend_score"] > 0.2:
        trend = "uptrend"
    elif latest["ema_50"] < latest["ema_200"] and latest["trend_score"] < -0.2:
        trend = "downtrend"
    else:
        trend = "sideways"

    # RSI signal
    rsi = latest["rsi_14"]
    if rsi > 70:
        rsi_signal = "overbought"
    elif rsi < 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    # MACD signal
    macd_signal = "bullish" if latest["macd"] > latest["macd_signal"] else "bearish"

    # Bollinger position
    bb_pct = latest["bb_pct_b"]
    if bb_pct > 0.8:
        bb_position = "near_upper_band"
    elif bb_pct < 0.2:
        bb_position = "near_lower_band"
    else:
        bb_position = "middle"

    # Sentiment proxy: ATR percentile (high ATR = high fear)
    atr_percentile = float(
        (df["atr_14"].dropna() <= latest["atr_14"]).mean()
    )
    if atr_percentile > 0.75:
        sentiment = "fearful"
    elif atr_percentile < 0.35:
        sentiment = "complacent"
    else:
        sentiment = "neutral"

    return {
        "date": latest.name.strftime("%Y-%m-%d"),
        "close": round(float(latest["Close"]), 2),
        "trend": trend,
        "trend_score": round(float(latest["trend_score"]), 3),
        "ema_50": round(float(latest["ema_50"]), 2),
        "ema_200": round(float(latest["ema_200"]), 2),
        "rsi_14": round(float(rsi), 2),
        "rsi_signal": rsi_signal,
        "macd": round(float(latest["macd"]), 4),
        "macd_signal_line": round(float(latest["macd_signal"]), 4),
        "macd_signal": macd_signal,
        "bb_pct_b": round(float(bb_pct), 3),
        "bb_position": bb_position,
        "atr_14": round(float(latest["atr_14"]), 2),
        "atr_percentile": round(atr_percentile, 3),
        "sentiment_proxy": sentiment,
    }


def get_sentiment_score(summary: dict) -> dict:
    """
    Consolidate the 5 indicator signals from get_market_summary() into a single
    weighted sentiment rating.

    Scoring:
        trend     — weight 2  (+1 uptrend, -1 downtrend,  0 sideways)
        macd      — weight 1  (+1 bullish,  -1 bearish)
        rsi       — weight 1  (+1 oversold, -1 overbought, 0 neutral)
        bb        — weight 1  (+1 near_lower, -1 near_upper, 0 middle)
        volatility— weight 1  (+1 complacent, -1 fearful,   0 neutral)

    Total range: -6 to +6
    Mapped to: BULLISH / CAUTIOUSLY_BULLISH / NEUTRAL / CAUTIOUSLY_BEARISH / BEARISH
    """
    # ── Individual scores ────────────────────────────────────────────────────
    trend_val = summary["trend"]
    trend_score = (
        2 if trend_val == "uptrend" else
        -2 if trend_val == "downtrend" else
        0
    )

    macd_score = 1 if summary["macd_signal"] == "bullish" else -1

    rsi_val = summary["rsi_signal"]
    rsi_score = (
        1 if rsi_val == "oversold" else
        -1 if rsi_val == "overbought" else
        0
    )

    bb_val = summary["bb_position"]
    bb_score = (
        1 if bb_val == "near_lower_band" else
        -1 if bb_val == "near_upper_band" else
        0
    )

    vol_val = summary["sentiment_proxy"]
    vol_score = (
        1 if vol_val == "complacent" else
        -1 if vol_val == "fearful" else
        0
    )

    total = trend_score + macd_score + rsi_score + bb_score + vol_score

    # ── Overall rating ───────────────────────────────────────────────────────
    if total >= 4:
        rating = "BULLISH"
    elif total >= 1:
        rating = "CAUTIOUSLY_BULLISH"
    elif total >= -1:
        rating = "NEUTRAL"
    elif total >= -3:
        rating = "CAUTIOUSLY_BEARISH"
    else:
        rating = "BEARISH"

    # ── Human-readable signal summary ────────────────────────────────────────
    parts = []
    if trend_val == "uptrend":
        parts.append("uptrend intact")
    elif trend_val == "downtrend":
        parts.append("downtrend in force")
    else:
        parts.append("no clear trend")

    parts.append(f"MACD {summary['macd_signal']}")
    parts.append(f"RSI {summary['rsi_14']:.1f} ({rsi_val})")

    if bb_val != "middle":
        parts.append(f"price {bb_val.replace('_', ' ')}")

    if vol_val != "neutral":
        parts.append(f"volatility {vol_val}")

    conviction = (
        "strong conviction" if abs(total) >= 4 else
        "moderate conviction" if abs(total) >= 2 else
        "low conviction"
    )
    signal_summary = ", ".join(parts) + f" — {conviction}"

    return {
        "overall_sentiment": rating,
        "total_score": total,
        "max_score": 6,
        "signal_summary": signal_summary,
        "signals": {
            "trend":      {"value": trend_val,                      "score": trend_score},
            "macd":       {"value": summary["macd_signal"],         "score": macd_score},
            "rsi":        {"value": f"{rsi_val} ({summary['rsi_14']:.1f})", "score": rsi_score},
            "bollinger":  {"value": bb_val,                         "score": bb_score},
            "volatility": {"value": vol_val,                        "score": vol_score},
        },
    }
