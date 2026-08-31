import MetaTrader5 as mt5
from time import sleep
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from reinforcement_learning.features import add_features

# connect to MetaTrader5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()
# check symbol
if not mt5.symbol_select("XAUUSD", True):
    print(f"error select {"XAUUSD"}:", mt5.last_error())
    mt5.shutdown()
    quit()


def get_candles():
    data = pd.DataFrame(mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 500))
    data["volume"] = data["tick_volume"]
    data.drop(columns=["real_volume", "spread", "tick_volume"], inplace=True)
    data['time'] = pd.to_datetime(data['time'], unit="s")
    data.columns = [c.capitalize() for c in data.columns]
    return add_features(data)

def get_current_pos():
    return mt5.positions_get(symbol="XAUUSD")

def get_obs(feat_df, feature_cols, position_state=None):
    row = feat_df.iloc[-1]
    market = row[feature_cols].astype(float).to_numpy(dtype=np.float32)
    if position_state is None:
        position_state = np.zeros(6, dtype=np.float32)
    obs = np.nan_to_num(np.concatenate([market, position_state]))
    return obs.reshape(1, -1)

def load_model():
    model = PPO.load("model/3_return_70%_1H/ppo_xauusd.zip", device="auto")
    vecnorm = VecNormalize.load("model/3_return_70%_1H/ppo_xauusd_vecorn.pkl", DummyVecEnv([lambda: None]))
    vecnorm.training = False
    return model, vecnorm

def place_order(direction, sl, tp, lot):
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    

# Main Logic
while True:
    df_feat, feature_cols = get_candles()
    print(df_feat)
    print(get_current_pos())
    sleep(1)

mt5.shutdown()
