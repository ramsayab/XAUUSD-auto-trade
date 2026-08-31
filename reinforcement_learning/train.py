from __future__ import annotations

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import cast
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features import add_features
from trading_env import BracketTradingEnv


DATA_PATH = ["data/XAUUSD_2004-2022.csv", "data/XAUUSD_2023-2026.csv"]
DECISION_TIMEFRAME = "15min"
RISK_FRACTION = 0.005
TRAIN_FRACTION = 0.8
TOTAL_TIMESTEPS = 1_000_000
EPISODE_STEPS = 2_048
INITIAL_EQUITY = 10_000


def load_data(path):
    data = pd.read_csv(path, parse_dates=["Date"])
    return data.sort_values("Date").set_index("Date")


def resample_ohlcv(data, timeframe=DECISION_TIMEFRAME):
    return data.resample(timeframe, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close"])


def slice_execution(execution_df, decision_df):
    return execution_df.loc[
        (execution_df.index > decision_df.index.min()) &
        (execution_df.index <= decision_df.index.max())
    ].copy()


def make_env(decision_df, execution_df, feature_cols, randomize_start=False):
    return Monitor(BracketTradingEnv(
        decision_df=decision_df,
        execution_df=execution_df,
        feature_cols=feature_cols,
        risk_fraction=RISK_FRACTION,
        randomize_start=randomize_start,
        initial_equity=INITIAL_EQUITY,
        max_episode_steps=EPISODE_STEPS if randomize_start else None,
    ))


class CaptureEpisode(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.saved_equity = pd.DataFrame()
        self.saved_trades = pd.DataFrame()
        self.saved_final_balance = None

    def step(self, action):
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        if terminated or truncated:
            bracket_env = cast(BracketTradingEnv, self.env)
            self.saved_equity = bracket_env.equity_curve().copy()
            self.saved_trades = bracket_env.trade_log().copy()
            self.saved_final_balance = float(bracket_env.equity)
        return result


def run_backtest(model, vec_env, capture_env):
    obs = vec_env.reset()
    actions_log = []
    done = np.array([False])
    while not done[0]:
        bracket_env = cast(BracketTradingEnv, capture_env.env)
        step = bracket_env.i
        price = float(bracket_env._row()["Close"])
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action[0], dtype=int)
        obs, _, done, _ = vec_env.step([action])
        actions_log.append({
            "index": step, "price": price, "direction": int(action[0]),
            "sl_bucket": int(action[1]), "tp_bucket": int(action[2]),
        })
    return (
        pd.DataFrame(actions_log),
        capture_env.saved_equity,
        capture_env.saved_trades,
        capture_env.saved_final_balance,
    )


def main():
    print("Loading source candles...")
    train_execution = load_data(DATA_PATH[0])
    test_execution = load_data(DATA_PATH[1])

    print(f"Resampling decision candles to {DECISION_TIMEFRAME}...")
    train_decision = resample_ohlcv(train_execution)
    test_decision = resample_ohlcv(test_execution)
    train_features, feature_cols = add_features(train_decision)
    test_features, _ = add_features(test_decision)

    split = int(len(train_features) * TRAIN_FRACTION)
    train_df = train_features.iloc[:split].copy()
    val_df = train_features.iloc[split:].copy()
    train_exec = slice_execution(train_execution, train_df)
    val_exec = slice_execution(train_execution, val_df)
    print(f"Train: {len(train_df):,} | Validation: {len(val_df):,} | Test: {len(test_features):,}")
    print(f"Features: {len(feature_cols)} | Risk per trade: {RISK_FRACTION:.2%} of equity")

    train_raw = make_env(train_df, train_exec, feature_cols, randomize_start=True)
    train_vec = VecNormalize(
        DummyVecEnv([lambda: train_raw]),
        norm_obs=True, norm_reward=True, clip_obs=10.0,
    )

    print("Training PPO...")
    model = PPO(
        "MlpPolicy", train_vec, device="cpu", verbose=1, seed=42,
        learning_rate=6e-5, gamma=0.99, gae_lambda=0.95,
        clip_range=0.1, ent_coef=0.02, n_steps=EPISODE_STEPS,
        batch_size=1024, n_epochs=5, target_kl=0.025,
        policy_kwargs={
            "net_arch": [256, 128],
            "optimizer_kwargs": {"weight_decay": 1e-5},
        },
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("model/ppo_xauusd")
    train_vec.save("model/ppo_xauusd_vecnorm.pkl")

    print("Evaluating the training split...")
    train_eval_raw = CaptureEpisode(BracketTradingEnv(
        train_df, train_exec, feature_cols,
        risk_fraction=RISK_FRACTION,
    ))
    train_eval_vec = VecNormalize(
        DummyVecEnv([lambda: train_eval_raw]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )
    train_eval_vec.obs_rms = train_vec.obs_rms
    _, train_equity, train_trades, train_balance = run_backtest(
        model, train_eval_vec, train_eval_raw
    )

    print("Backtesting on the unseen test set...")
    test_raw = CaptureEpisode(BracketTradingEnv(
        test_features, test_execution, feature_cols,
        risk_fraction=RISK_FRACTION,
    ))
    test_vec = VecNormalize(
        DummyVecEnv([lambda: Monitor(test_raw)]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )
    test_vec.obs_rms = train_vec.obs_rms
    log_df, test_equity, test_trades, test_balance = run_backtest(
        model, test_vec, test_raw
    )

    plt.figure(figsize=(16, 6))
    plt.plot(log_df["index"], log_df["price"], color="black", linewidth=1)
    longs = log_df[log_df["direction"] == 1]
    shorts = log_df[log_df["direction"] == 2]
    plt.scatter(longs["index"], longs["price"], marker="^", color="green", s=50, label="Long")
    plt.scatter(shorts["index"], shorts["price"], marker="v", color="red", s=50, label="Short")
    plt.title("XAUUSD PPO decisions")
    plt.xlabel("Decision bar")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig("xauusd_decisions.png", dpi=150)
    plt.close()

    # Plot training equity history with legend for trade counts
    if not train_equity.empty:
        if not train_trades.empty and "direction" in train_trades.columns:
            train_long = len(train_trades[train_trades["direction"] == 1])
            train_short = len(train_trades[train_trades["direction"] == -1])
        else:
            train_long = 0
            train_short = 0
        plt.figure(figsize=(12, 6))
        plt.plot(
            train_equity["time"] if "time" in train_equity.columns else train_equity.index, 
            train_equity["equity"], 
            color="blue", 
            linewidth=1.5,
            label=f"Equity (Long Trades: {train_long}, Short Trades: {train_short})"
        )
        plt.title("PPO Training Equity Curve")
        plt.xlabel("Date / Time" if "time" in train_equity.columns else "Steps")
        plt.ylabel("Equity ($)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("train_equity.png", dpi=150)
        plt.close()

    # Plot testing equity history with legend for trade counts
    if not test_equity.empty:
        if not test_trades.empty and "direction" in test_trades.columns:
            test_long = len(test_trades[test_trades["direction"] == 1])
            test_short = len(test_trades[test_trades["direction"] == -1])
        else:
            test_long = 0
            test_short = 0
        plt.figure(figsize=(12, 6))
        plt.plot(
            test_equity["time"] if "time" in test_equity.columns else test_equity.index, 
            test_equity["equity"], 
            color="green", 
            linewidth=1.5,
            label=f"Equity (Long Trades: {test_long}, Short Trades: {test_short})"
        )
        plt.title("PPO Testing Equity Curve")
        plt.xlabel("Date / Time" if "time" in test_equity.columns else "Steps")
        plt.ylabel("Equity ($)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("test_equity.png", dpi=150)
        plt.close()

    # Plot trade holding durations distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train Durations Distribution
    if not train_trades.empty and "entry_time" in train_trades.columns:
        train_durations = (pd.to_datetime(train_trades["exit_time"]) - pd.to_datetime(train_trades["entry_time"])).dt.total_seconds() / 3600.0
        axes[0].hist(train_durations, bins=30, color="royalblue", alpha=0.7, edgecolor="black")
        axes[0].set_title(f"Train Trade Durations (N={len(train_trades)})")
        axes[0].set_xlabel("Holding Time (Hours)")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        # Add stats lines
        mean_val = train_durations.mean()
        median_val = train_durations.median()
        axes[0].axvline(mean_val, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_val:.1f}h")
        axes[0].axvline(median_val, color="purple", linestyle="dashed", linewidth=1.5, label=f"Median: {median_val:.1f}h")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No Train Trades", ha="center", va="center")
        
    # Test Durations Distribution
    if not test_trades.empty and "entry_time" in test_trades.columns:
        test_durations = (pd.to_datetime(test_trades["exit_time"]) - pd.to_datetime(test_trades["entry_time"])).dt.total_seconds() / 3600.0
        axes[1].hist(test_durations, bins=30, color="seagreen", alpha=0.7, edgecolor="black")
        axes[1].set_title(f"Test Trade Durations (N={len(test_trades)})")
        axes[1].set_xlabel("Holding Time (Hours)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        # Add stats lines
        mean_val = test_durations.mean()
        median_val = test_durations.median()
        axes[1].axvline(mean_val, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_val:.1f}h")
        axes[1].axvline(median_val, color="purple", linestyle="dashed", linewidth=1.5, label=f"Median: {median_val:.1f}h")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No Test Trades", ha="center", va="center")
        
    plt.suptitle("Distribution of Trade Holding Times (Hours)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("trade_durations.png", dpi=150)
    plt.close()

    # Plot trade durations comparison (Long vs Short)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Train Comparison
    if not train_trades.empty and "entry_time" in train_trades.columns:
        train_df_trades = train_trades.copy()
        train_df_trades["duration_hours"] = (pd.to_datetime(train_df_trades["exit_time"]) - pd.to_datetime(train_df_trades["entry_time"])).dt.total_seconds() / 3600.0
        
        long_durations = train_df_trades[train_df_trades["direction"] == 1]["duration_hours"]
        short_durations = train_df_trades[train_df_trades["direction"] == -1]["duration_hours"]
        
        data_to_plot = []
        labels = []
        if not long_durations.empty:
            data_to_plot.append(long_durations)
            labels.append(f"Long (N={len(long_durations)})")
        if not short_durations.empty:
            data_to_plot.append(short_durations)
            labels.append(f"Short (N={len(short_durations)})")
            
        if data_to_plot:
            bp = axes[0].boxplot(data_to_plot, patch_artist=True, showmeans=True)
            axes[0].set_xticklabels(labels)
            colors = ["#1f77b4", "#ff7f0e"] # blue and orange
            for patch, color in zip(bp["boxes"], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            for mean in bp["means"]:
                mean.set(marker="o", markerfacecolor="yellow", markeredgecolor="black", markersize=6)
            axes[0].set_title("Train: Long vs Short Holding Time")
            axes[0].set_ylabel("Holding Time (Hours)")
            axes[0].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[0].text(0.5, 0.5, "No Long/Short Trades", ha="center", va="center")
    else:
        axes[0].text(0.5, 0.5, "No Train Trades", ha="center", va="center")
        
    # Test Comparison
    if not test_trades.empty and "entry_time" in test_trades.columns:
        test_df_trades = test_trades.copy()
        test_df_trades["duration_hours"] = (pd.to_datetime(test_df_trades["exit_time"]) - pd.to_datetime(test_df_trades["entry_time"])).dt.total_seconds() / 3600.0
        
        long_durations = test_df_trades[test_df_trades["direction"] == 1]["duration_hours"]
        short_durations = test_df_trades[test_df_trades["direction"] == -1]["duration_hours"]
        
        data_to_plot = []
        labels = []
        if not long_durations.empty:
            data_to_plot.append(long_durations)
            labels.append(f"Long (N={len(long_durations)})")
        if not short_durations.empty:
            data_to_plot.append(short_durations)
            labels.append(f"Short (N={len(short_durations)})")
            
        if data_to_plot:
            bp = axes[1].boxplot(data_to_plot, patch_artist=True, showmeans=True)
            axes[1].set_xticklabels(labels)
            colors = ["#2ca02c", "#d62728"] # green and red
            for patch, color in zip(bp["boxes"], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            for mean in bp["means"]:
                mean.set(marker="o", markerfacecolor="yellow", markeredgecolor="black", markersize=6)
            axes[1].set_title("Test: Long vs Short Holding Time")
            axes[1].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[1].text(0.5, 0.5, "No Long/Short Trades", ha="center", va="center")
    else:
        axes[1].text(0.5, 0.5, "No Test Trades", ha="center", va="center")
        
    plt.suptitle("Holding Time Comparison: Long vs Short Trades (Hours)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("duration_comparison.png", dpi=150)
    plt.close()


    for label, equity, trades in (
        ("Train", train_equity, train_trades),
        ("Test", test_equity, test_trades),
    ):
        final_balance = train_balance if label == "Train" else test_balance
        if equity.empty or final_balance is None:
            print(f"{label} final balance: unavailable (no completed episode)")
            continue
        total_return = (float(final_balance) / float(INITIAL_EQUITY) - 1) * 100
        print(
            f"{label} final balance: {float(final_balance):.2f} | "
            f"Return: {total_return:.2f}% | Trades: {len(trades)}"
        )

if __name__ == "__main__":
    main()