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
    out["ema_50"] = ta.trend.EMAIndicator(close=out["Close"], window=50).ema_indicator()
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

    # Daily return
    out["daily_return"] = out["Close"].pct_change()

    return out


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
