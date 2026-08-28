from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class Position:
    direction: int = 0
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    units: float = 0.0
    risk_cash: float = 0.0
    sl_distance: float = 0.0
    tp_r: float = 0.0
    sl_atr_mult: float = 0.0
    bars_in_trade: int = 0
    entry_time: Optional[pd.Timestamp] = None


class BracketTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, decision_df, execution_df, feature_cols,
        sl_atr_multipliers=(1.0, 1.5, 2.0),
        tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        initial_equity=10_000.0, risk_fraction=0.005,
        spread_price=0.20, slippage_price=0.02,
        commission_per_trade=0.0, holding_penalty=0.00002,
        reward_mtm_weight=0.01, max_episode_steps=None, randomize_start=False,
    ):
        super().__init__()
        self.decision_df = decision_df.dropna(subset=feature_cols + ["atr"]).copy()
        self.execution_df = execution_df.sort_index().copy()
        self.feature_cols = list(feature_cols)
        self.sl_atr_multipliers = tuple(sl_atr_multipliers)
        self.tp_r_multipliers = tuple(tp_r_multipliers)
        self.initial_equity = float(initial_equity)
        self.risk_fraction = float(risk_fraction)
        self.spread_price = float(spread_price)
        self.slippage_price = float(slippage_price)
        self.commission_per_trade = float(commission_per_trade)
        self.holding_penalty = float(holding_penalty)
        self.reward_mtm_weight = float(reward_mtm_weight)
        self.max_episode_steps = max_episode_steps or max(len(self.decision_df) - 2, 1)
        self.randomize_start = randomize_start

        if self.risk_fraction <= 0:
            raise ValueError("risk_fraction must be positive")
        if len(self.decision_df) < 3:
            raise ValueError("decision_df must contain at least three decision bars")

        self._execution_high = self.execution_df["High"].to_numpy(dtype=np.float64)
        self._execution_low = self.execution_df["Low"].to_numpy(dtype=np.float64)
        self._execution_index = self.execution_df.index
        self.action_space = spaces.MultiDiscrete([
            3, len(self.sl_atr_multipliers), len(self.tp_r_multipliers)
        ])
        self.n_position_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_cols) + self.n_position_features,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_start:
            max_start = max(len(self.decision_df) - self.max_episode_steps - 1, 1)
            self.i = int(self.np_random.integers(0, max_start))
        else:
            self.i = 0
        self.steps = 0
        self.equity = self.initial_equity
        self.position = Position()
        self.history = []
        self.trades = []
        return self._observation(), {}

    def _row(self):
        return self.decision_df.iloc[self.i]

    def _time(self):
        return self.decision_df.index[self.i]

    def _next_time(self):
        return self.decision_df.index[min(self.i + 1, len(self.decision_df) - 1)]

    def _observation(self):
        row = self._row()
        close = float(row["Close"])
        atr = max(float(row["atr"]), 1e-12)
        p = self.position
        if p.direction == 0:
            position_state = np.zeros(6, dtype=np.float32)
        else:
            unrealized = (close - p.entry_price) * p.units * p.direction
            position_state = np.array([
                p.direction, unrealized / max(p.risk_cash, 1e-12),
                min(p.bars_in_trade / 100.0, 10.0),
                ((p.tp - close) * p.direction) / atr,
                ((close - p.sl) * p.direction) / atr, p.tp_r,
            ], dtype=np.float32)
        market = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        return np.nan_to_num(np.concatenate([market, position_state]), nan=0.0, posinf=10.0, neginf=-10.0)

    def _entry_price(self, close, direction):
        return close + direction * (self.spread_price / 2 + self.slippage_price)

    def _exit_price(self, price, direction):
        return price - direction * (self.spread_price / 2 + self.slippage_price)

    def _open_position(self, direction, sl_idx, tp_idx):
        row = self._row()
        entry = self._entry_price(float(row["Close"]), direction)
        sl_distance = max(self.sl_atr_multipliers[sl_idx] * float(row["atr"]), 1e-8)
        tp_r = self.tp_r_multipliers[tp_idx]
        risk_cash = max(self.equity * self.risk_fraction, 1e-8)
        units = risk_cash / sl_distance
        self.position = Position(
            direction=direction, entry_price=entry,
            sl=entry - direction * sl_distance,
            tp=entry + direction * tp_r * sl_distance,
            units=units, risk_cash=risk_cash,
            sl_distance=sl_distance, tp_r=tp_r,
            sl_atr_mult=self.sl_atr_multipliers[sl_idx], entry_time=self._time(),
        )

    def _close_position(self, raw_price, reason):
        p = self.position
        if p.direction == 0:
            return
        exit_price = self._exit_price(raw_price, p.direction)
        pnl = (exit_price - p.entry_price) * p.units * p.direction
        pnl -= self.commission_per_trade
        self.equity += pnl
        self.trades.append({
            "entry_time": p.entry_time, "exit_time": self._time(),
            "direction": p.direction, "entry_price": p.entry_price,
            "exit_price": exit_price, "units": p.units,
            "risk_cash": p.risk_cash,
            "r_mult": pnl / max(p.risk_cash, 1e-12),
            "pnl": pnl, "exit_reason": reason,
        })
        self.position = Position()

    def _simulate_execution(self):
        if self.position.direction == 0:
            return
        lo = int(self._execution_index.searchsorted(self._time(), side="right"))
        hi = int(self._execution_index.searchsorted(self._next_time(), side="right"))
        for idx in range(lo, hi):
            p = self.position
            if p.direction == 0:
                break
            sl_hit = self._execution_low[idx] <= p.sl if p.direction == 1 else self._execution_high[idx] >= p.sl
            tp_hit = self._execution_high[idx] >= p.tp if p.direction == 1 else self._execution_low[idx] <= p.tp
            if sl_hit:
                self._close_position(p.sl, "SL")
                break
            if tp_hit:
                self._close_position(p.tp, "TP")
                break

    def step(self, action):
        direction_raw, sl_idx, tp_idx = np.asarray(action, dtype=int)
        desired_direction = {0: 0, 1: 1, 2: -1}[int(direction_raw)]
        previous_equity = self.equity
        close = float(self._row()["Close"])

        if self.position.direction != 0:
            if desired_direction == 0:
                self._close_position(close, "manual_close")
            elif desired_direction != self.position.direction:
                self._close_position(close, "flip_close")
                self._open_position(desired_direction, int(sl_idx), int(tp_idx))
        elif desired_direction != 0:
            self._open_position(desired_direction, int(sl_idx), int(tp_idx))

        self._simulate_execution()
        if self.position.direction != 0:
            self.position.bars_in_trade += 1
            unrealized = (close - self.position.entry_price) * self.position.units * self.position.direction
        else:
            unrealized = 0.0

        reward_unit = max(previous_equity * self.risk_fraction, 1e-12)
        reward = (self.equity - previous_equity) / reward_unit
        if self.position.direction != 0:
            reward += unrealized / max(self.position.risk_cash, 1e-12) * self.reward_mtm_weight
            reward -= self.holding_penalty

        self.history.append({"time": self._time(), "equity": self.equity,
                             "position": self.position.direction, "close": close,
                             "reward": reward})
        self.i += 1
        self.steps += 1
        terminated = self.i >= len(self.decision_df) - 2
        truncated = self.steps >= self.max_episode_steps
        if terminated or truncated:
            if self.position.direction != 0:
                self._close_position(float(self.decision_df.iloc[self.i]["Close"]), "episode_end")
            observation_shape = self.observation_space.shape
            assert observation_shape is not None
            observation = np.zeros(observation_shape, dtype=np.float32)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, truncated, {
            "equity": self.equity, "position": self.position.direction,
            "n_trades": len(self.trades),
        }

    def equity_curve(self):
        return pd.DataFrame(self.history)

    def trade_log(self):
        return pd.DataFrame(self.trades)