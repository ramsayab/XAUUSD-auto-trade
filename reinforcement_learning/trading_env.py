import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=20, initial_balance=10_000,
                 transaction_cost=0.0002, decision_interval=60):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # decision_interval=60 -> 1 Hour
        self.decision_interval = decision_interval

        self.feature_cols = [c for c in df.columns if c != "Date"]

        n_features = len(self.feature_cols)
        obs_dim = window_size * n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
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

    def _compute_equity(self, price):
        if self.position == 0:
            return self.balance
        unrealized_pct = (price - self.entry_price) / self.entry_price * self.position
        return self.balance * (1 + unrealized_pct)

    def _apply_action(self, action, price):
        """Eksekusi keputusan agent (buka/tutup posisi). Dipanggil SEKALI per blok keputusan."""
        if action == 1:  # BUY
            if self.position == -1:
                self._close_position(price)
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost

        elif action == 2:  # SELL
            if self.position == 1:
                self._close_position(price)
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.balance -= self.balance * self.transaction_cost
        # action == 0 (HOLD) -> tidak melakukan apa-apa

    def step(self, action):
        equity_before = self._compute_equity(self._current_price())

        # === 1. Terapkan keputusan agent SEKALI di awal blok ===
        self._apply_action(action, self._current_price())

        terminated = False
        for _ in range(self.decision_interval):
            self.current_step += 1
            if self.current_step >= len(self.df) - 1:
                terminated = True
                break

        price = self._current_price()

        # Tutup posisi otomatis kalau episode berakhir
        if terminated and self.position != 0:
            self._close_position(price)

        current_equity = self._compute_equity(price)

        # Reward = perubahan equity relatif selama satu blok keputusan ini
        # (mencakup transaction cost + PnL realized/unrealized, sekali hitung)
        reward = (current_equity / equity_before) - 1.0

        self.equity_curve.append(current_equity)

        if not terminated:
            obs = self._get_obs()
        else:
            obs_shape = self.observation_space.shape
            assert obs_shape is not None
            obs = np.zeros(obs_shape, dtype=np.float32)

        truncated = False
        info = {"balance": self.balance, "position": self.position, "price": price}

        return obs, float(reward), terminated, truncated, info

    def _close_position(self, price):
        """Realisasi PnL saat menutup posisi. Update balance langsung."""
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        self.balance += self.balance * pnl_pct
        self.balance -= self.balance * self.transaction_cost
        self.position = 0
        self.entry_price = 0.0

    def render(self):
        print(f"Step {self.current_step} | Balance: {self.balance:.2f} | "
              f"Position: {self.position}")