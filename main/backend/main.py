import os
import sys
import threading
from contextlib import asynccontextmanager
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from reinforcement_learning.features import add_features
from reinforcement_learning.trading_env import BracketTradingEnv

from db import model, schemas
from db.database import get_db, create_tables, SessionLocal
from datetime import datetime

create_tables()

state = {"running": True,
         "entry_time": datetime.now()}

""""""
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

    model = PPO.load("model/2/ppo.zip", device="cpu")
    vecnorm = VecNormalize.load(
        "model/2/ppo_vecnorm.pkl", DummyVecEnv([make_dummy_env]))
    vecnorm.training = False
    return model, vecnorm, feature_cols

def add_history(data: schemas.HistoryBase):
    db = SessionLocal()
    try:
        db_history = model.History(**data.model_dump())
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        return schemas.HistoryResponse.model_validate(db_history)
    finally:
        db.close()

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
    state["entry_time"] = datetime.now()
    return result

def close_position():
    pos = mt5.positions_get(symbol="XAUUSD")
    if not pos: return None
    tick = mt5.symbol_info_tick("XAUUSD")
    order = ""
    if pos[0].type == mt5.POSITION_TYPE_BUY:
        ordertype = mt5.ORDER_TYPE_SELL
        order = "Long"
        price = tick.bid
    else:
        ordertype = mt5.ORDER_TYPE_BUY
        order = "Short"
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
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        add_history(
            schemas.HistoryBase(
                order_type = order,
                profit = pos[0].profit,
                start_time=state["entry_time"],
                end_time=datetime.now()
            )
        )
    return result
""""""

def trade_loop():
    if not mt5.initialize():
        print("MT5 init Failed")
        return
    model, vecnorm, feature_cols = load_model()

    while state["running"]:
        df_feat, feature_cols = get_candles()
        obs_norm = vecnorm.normalize_obs(get_obs(df_feat, feature_cols))
        action, _ = model.predict(obs_norm, deterministic=True)
        direction, sl_idx, tp_idx = action[0]
    
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
                place_order(dir_sign, sl_price, tp_price, lot=0.5)
            else:
                print("bias is still the same, no change")
        elif direction == 0 and pos != 0:
            close_position()
        time.sleep(1)



@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=trade_loop, daemon=True)
    thread.start()
    yield
    state["running"] = False
    mt5.shutdown()

app = FastAPI(debug=True, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



@app.get("/current_pos")
def get_pos():
    pos = mt5.positions_get(symbol="XAUUSD")
    if not pos: return None
    return {
        "lot": pos[0].volume,
        "entry_price": pos[0].price_open,
        "current_price": pos[0].price_current,
        "profit": pos[0].profit,
        "order_type": "Long" if pos[0].type == mt5.POSITION_TYPE_BUY else "Short",
        "balance": mt5.account_info().balance
    }

@app.get("/history_pos", response_model=List[schemas.HistoryResponse])
def get_history_pos(db: Session=Depends(get_db)):
    return db.query(model.History).offset(0).limit(100).all()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)