import MetaTrader5 as mt5
from time import sleep
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from reinforcement_learning.features import add_features
from reinforcement_learning.trading_env import BracketTradingEnv


def get_candles():
    data = pd.DataFrame(mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 500))
    data["volume"] = data["tick_volume"]
    data.drop(columns=["real_volume", "spread", "tick_volume"], inplace=True)
    data['time'] = pd.to_datetime(data['time'], unit="s")
    data = data.set_index("time")
    data.columns = [c.capitalize() for c in data.columns]
    return add_features(data)

def get_current_pos():
    pos = mt5.positions_get(symbol="XAUUSD")
    if not pos: return 0
    if pos[0].type == mt5.POSITION_TYPE_BUY:
        return 1
    elif pos[0].type == mt5.POSITION_TYPE_SELL:
        return -1

def get_obs(feat_df, feature_cols, position_state=None):
    row = feat_df.iloc[-1]
    market = row[feature_cols].astype(float).to_numpy(dtype=np.float32)
    if position_state is None:
        position_state = np.zeros(6, dtype=np.float32)
    obs = np.nan_to_num(np.concatenate([market, position_state]))
    return obs.reshape(1, -1)

def load_model():
    df_feat, feature_cols = get_candles()
    dummy_decision = df_feat.iloc[-10:]
    dummy_exec = dummy_decision
    def make_dummy_env():
        return BracketTradingEnv(
            decision_df=dummy_decision,
            execution_df=dummy_exec,
            feature_cols=feature_cols,
        )

    model = PPO.load("model/3_return_70%_1H/ppo_xauusd.zip", device="cpu")
    vecnorm = VecNormalize.load(
        "model/3_return_70%_1H/ppo_xauusd_vecnorm.pkl", DummyVecEnv([make_dummy_env]))
    vecnorm.training = False
    return model, vecnorm, feature_cols

def place_order(direction, sl, tp, lot):
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick("XAUUSD").ask if direction == 1 else mt5.symbol_info_tick("XAUUSD").bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10, "magic": 123456,
        "comment": "ppo_xauusd",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(result)
    return result

def close_position():
    pos = mt5.positions_get(symbol="XAUUSD")
    if not pos: return None
    tick = mt5.symbol_info_tick("XAUUSD")
    if pos[0].type == mt5.POSITION_TYPE_BUY:
        ordertype = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        ordertype = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "volume": pos[0].volume,
        "type": ordertype,
        "price": price,
        "position": pos[0].ticket,
        "deviation": 10, "magic": 123456,
        "comment": "close_xauusd",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(result)
    return result



# connect to MetaTrader5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()
# check symbol
if not mt5.symbol_select("XAUUSD", True):
    print(f"error select {"XAUUSD"}:", mt5.last_error())
    mt5.shutdown()
    quit()

model, vecnorm, feature_cols = load_model()

# Main Logic
while True:
    df_feat, feature_cols = get_candles()
    obs_norm = vecnorm.normalize_obs(get_obs(df_feat, feature_cols))
    action, _ = model.predict(obs_norm, deterministic=True)
    direction, sl_idx, tp_idx = action[0]
    print(f"directon={direction}\nsl={sl_idx}\ntp={tp_idx}")

    pos = get_current_pos()
    if direction != 0:
        sl_mults = (1.0, 1.5, 2.0)
        tp_mults = (1.0, 1.5, 2.0, 3.0)
        atr = float(df_feat.iloc[-1]["atr"])
        close = float(df_feat.iloc[-1]["Close"])
        dir_sign = 1 if direction == 1 else -1
        if pos != dir_sign:
            close_position()
            sl_dist = sl_mults[sl_idx] * atr
            sl_price = close - dir_sign * sl_dist
            tp_price = close + dir_sign * tp_mults[tp_idx] * sl_dist
            place_order(dir_sign, sl_price, tp_price, lot=0.1)
        else:
            print("bias is still the same, no change")
    elif direction == 0 and pos != 0:
        close_position()
    print(f"\n\n\n\n\n\nCurrent position: {pos}")
    print(f"Current equity: {mt5.account_info().balance}")
    print("=" * 25)
    sleep(5)
mt5.shutdown()
