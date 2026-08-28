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
DECISION_TIMEFRAME = "1h"
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
        "MlpPolicy", train_vec, device="cuda", verbose=1, seed=42,
        learning_rate=6e-5, gamma=0.99, gae_lambda=0.95,
        clip_range=0.1, ent_coef=0.03, n_steps=EPISODE_STEPS,
        batch_size=512, n_epochs=5, target_kl=0.025,
        policy_kwargs={
            "net_arch": [128, 64],
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