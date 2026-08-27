from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)


def add_features(
    df: pd.DataFrame, atr_period: int = 14, warmup_bars: int = 250
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["atr"] = true_range(out).ewm(
        alpha=1 / atr_period, adjust=False, min_periods=atr_period
    ).mean()
    out["ema20"] = out["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = out["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["Close"].ewm(span=200, adjust=False, min_periods=200).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()
    out["rsi14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50.0)

    ema12 = out["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    bb_width = 4 * out["Close"].rolling(20, min_periods=20).std()

    eps = 1e-12
    atr = out["atr"].replace(0, np.nan)
    out["close_ema20_atr"] = (out["Close"] - out["ema20"]) / (atr + eps)
    out["close_ema50_atr"] = (out["Close"] - out["ema50"]) / (atr + eps)
    out["close_ema200_atr"] = (out["Close"] - out["ema200"]) / (atr + eps)
    out["ema20_ema50_atr"] = (out["ema20"] - out["ema50"]) / (atr + eps)
    out["macd_hist_atr"] = out["macd_hist"] / (atr + eps)
    out["roc20_atr"] = (out["Close"] - out["Close"].shift(20)) / (atr + eps)
    out["atr_close"] = out["atr"] / out["Close"]
    atr_fast = out["atr"].ewm(span=5, adjust=False, min_periods=5).mean()
    atr_slow = out["atr"].ewm(span=30, adjust=False, min_periods=30).mean()
    out["atr_fast_slow"] = atr_fast / atr_slow.replace(0, np.nan)
    out["bb_width_close"] = bb_width / out["Close"]
    candle_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["range_atr"] = (out["High"] - out["Low"]) / (atr + eps)
    out["upper_wick_ratio"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / candle_range
    out["lower_wick_ratio"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / candle_range

    for period in range(1, 6):
        out[f"ret{period}_atr"] = (out["Close"] - out["Close"].shift(period)) / (atr + eps)

    index = pd.DatetimeIndex(out.index)
    minute_of_day = index.hour * 60 + index.minute
    day_of_week = index.dayofweek
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    utc_index = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    utc_hour = utc_index.hour
    out["session_asia"] = ((utc_hour >= 0) & (utc_hour < 7)).astype(int)
    out["session_london"] = ((utc_hour >= 7) & (utc_hour < 16)).astype(int)
    out["session_newyork"] = ((utc_hour >= 13) & (utc_hour < 22)).astype(int)
    out["session_london_ny_overlap"] = ((utc_hour >= 13) & (utc_hour < 16)).astype(int)

    feature_cols = [
        "close_ema20_atr", "close_ema50_atr", "close_ema200_atr", "ema20_ema50_atr",
        "macd_hist_atr", "roc20_atr", "atr_close", "atr_fast_slow", "bb_width_close",
        "range_atr", "upper_wick_ratio", "lower_wick_ratio",
        "ret1_atr", "ret2_atr", "ret3_atr", "ret4_atr", "ret5_atr",
        "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        "session_asia", "session_london", "session_newyork", "session_london_ny_overlap",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out = out.iloc[warmup_bars:]
    return out.dropna(subset=feature_cols + ["Close", "atr"]), feature_cols