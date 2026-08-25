"""
features.py
Feature engineering sederhana & causal (hanya pakai data masa lalu).
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Return log harga (stationary, lebih stabil untuk ML dibanding harga mentah)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving averages (trend)
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_30"] = df["Close"].rolling(30).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_30"] - 1  # normalisasi jadi rasio, bukan harga absolut

    # Volatility (ATR sederhana)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]  # normalisasi ATR ke persentase harga

    # RSI sederhana
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50  # normalisasi ke [-1, 1]

    # Candle shape
    df["body_pct"] = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)

    # Buang kolom mentah yang tidak stationary agar tidak dipakai langsung sebagai fitur
    feature_df = df[[
        "Date", "Close",  # Close disimpan untuk hitung reward, bukan sebagai fitur "mentah" utama
        "log_return", "sma_ratio", "atr_pct", "rsi_norm", "body_pct",
    ]].copy()

    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df