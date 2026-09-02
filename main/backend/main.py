from fastapi import FastAPI, Depends, HTTPException
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
from db.database import get_db, create_tables

create_tables()

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/current_pos")
def get_current_pos():
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)