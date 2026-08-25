"""
trading_env.py
Environment trading sederhana untuk RL (Gymnasium-compatible).

Konsep dasar:
- Observation: window harga/indikator N candle terakhir + posisi saat ini
- Action: 0 = Hold, 1 = Buy, 2 = Sell (posisi selalu di-flat dulu sebelum ganti arah)
- Reward: perubahan equity per step (net dari biaya transaksi)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=20, initial_balance=10_000,
                 transaction_cost=0.0002):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # persentase per transaksi (spread+komisi disederhanakan)

        # Kolom fitur yang dipakai sebagai observation (selain harga close mentah)
        self.feature_cols = [c for c in df.columns if c not in ("Date",)]

        n_features = len(self.feature_cols)
        # Observation: window_size baris fitur (flatten) + posisi saat ini (1 nilai: -1, 0, 1)
        obs_dim = window_size * n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0          # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.equity_curve = [float(self.initial_balance)]

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.df.loc[
            self.current_step - self.window_size: self.current_step - 1,
            self.feature_cols,
        ].values.astype(np.float32).flatten()
        obs = np.concatenate([window, [float(self.position)]])
        return obs

    def _current_price(self):
        return float(self.df.loc[self.current_step, "Close"])

    def step(self, action):
        price = self._current_price()
        reward = 0.0

        # === Eksekusi aksi ===
        if action == 1:  # BUY
            if self.position == -1:
                reward += self._close_position(price)  # tutup short dulu
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost

        elif action == 2:  # SELL
            if self.position == 1:
                reward += self._close_position(price)  # tutup long dulu
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost
        # action == 0 (HOLD) -> tidak melakukan apa-apa

        # === Mark-to-market unrealized PnL untuk reward per-step ===
        if self.position != 0:
            unrealized = (price - self.entry_price) * self.position
            reward += unrealized / self.entry_price  # reward relatif, bukan cash mentah

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Tutup posisi otomatis di akhir episode
        if terminated and self.position != 0:
            reward += self._close_position(self._current_price())

        current_equity = self.balance + (
            (price - self.entry_price) * self.position * self.balance / price
            if self.position != 0 else 0
        )
        self.equity_curve.append(current_equity)

        if not terminated:
            obs = self._get_obs()
        else:
            obs_shape = self.observation_space.shape
            assert obs_shape is not None
            obs = np.zeros(obs_shape, dtype=np.float32)
        info = {"balance": self.balance, "position": self.position, "price": price}

        return obs, float(reward), terminated, truncated, info

    def _close_position(self, price):
        """Realisasi PnL saat menutup posisi. Return reward tambahan dari realisasi."""
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        self.balance += self.balance * pnl_pct
        self.balance -= self.balance * self.transaction_cost
        self.position = 0
        self.entry_price = 0.0
        return pnl_pct

    def render(self):
        print(f"Step {self.current_step} | Balance: {self.balance:.2f} | "
              f"Position: {self.position}")