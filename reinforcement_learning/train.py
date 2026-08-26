import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from features import add_features
from trading_env import SimpleTradingEnv

from collections import Counter
actions_taken = []


DATA_PATH = "data/XAUUSD_2004-2022.csv"
WINDOW_SIZE = 20
TRAIN_RATIO = 0.8
TOTAL_TIMESTEPS = 50_000  # naikkan (misal 200_000+) untuk hasil lebih matang


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def main():
    # 1. Load & feature engineering
    print("Loading data...")
    raw_df = load_data(DATA_PATH)
    feat_df = add_features(raw_df)
    print(f"Data setelah feature engineering: {len(feat_df)} baris")

    # 2. Split train/test secara temporal (TIDAK diacak, karena time-series)
    split_idx = int(len(feat_df) * TRAIN_RATIO)
    train_df = feat_df.iloc[:split_idx].reset_index(drop=True)
    test_df = feat_df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train: {len(train_df)} baris | Test: {len(test_df)} baris")

    # 3. Buat environment training
    train_env = DummyVecEnv([lambda: SimpleTradingEnv(train_df, window_size=WINDOW_SIZE, decision_interval=60)])

    # 4. Training PPO
    print("Training PPO...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        device='cpu'
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_xauusd_simple")
    print("Model tersimpan sebagai ppo_xauusd_simple.zip")

    # 5. Backtest di test set (data yang belum pernah dilihat model)
    print("Backtesting di test set...")
    test_env = SimpleTradingEnv(test_df, window_size=WINDOW_SIZE)
    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action))
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated

    print(Counter(actions_taken))

    # 6. Hasil & plot equity curve
    equity = test_env.equity_curve
    total_return = (equity[-1] / equity[0] - 1) * 100
    print(f"\nEquity awal : {equity[0]:.2f}")
    print(f"Equity akhir: {equity[-1]:.2f}")
    print(f"Total return: {total_return:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title(f"Equity Curve (Test Set) - Return: {total_return:.2f}%")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    print("Chart tersimpan sebagai equity_curve.png")


if __name__ == "__main__":
    main()