# XAUUSD Auto Trade

An automated trading system for the **XAUUSD** instrument using reinforcement learning with the PPO (_Proximal Policy Optimization_) algorithm. This project includes a simulation environment, training and backtesting workflows, MetaTrader 5 integration, a FastAPI REST API, and a React dashboard for monitoring positions and transaction history.

> [!WARNING]
> **Do not use a live account or real funds. Use a demo account only.** This project is experimental and **is not financial advice or an investment recommendation**. Trading carries the risk of losing some or all of your capital. The model may generate incorrect signals, fail to handle new market conditions, or execute orders with results that differ from the simulation.
>
> Before running the code, make sure MetaTrader 5 is connected to a **demo account**, trading permissions and lot sizes have been checked, and the system is continuously monitored. Users are fully responsible for their MetaTrader 5 configuration, account credentials, the decision to enable order execution, and any losses, interruptions, or other issues arising from the use of this code. The repository owner and contributors do not guarantee the system's profitability, availability, accuracy, or security.

### Demo video

<video src="https://github.com/user-attachments/assets/aac38bc7-0ba6-4b77-9803-018029225a6a" controls muted autoplay loop playsinline width="100%"></video>

## Training and Backtesting Results

<<<<<<< HEAD
The following charts show the evaluation results for the training and testing processes.
=======
berikut grafik evaluasi proses training dan testing.
>>>>>>> fc6edff450e12d0ed454adb9d6ae63a1f72de8f7

### Training Equity

![Training equity curve](model/3/train_equity.png)

### Testing Equity

![Testing equity curve](model/3/test_equity.png)

### Trade Duration

![Trade duration distribution](model/3/trade_durations.png)

The much higher training ROI is due to its coverage of 19 years (`2004-2022`), while testing covers 4 years (`2023-2026`). Therefore, the training ROI is a cumulative return over a longer period with more trading opportunities, so it cannot be concluded that the model is overfitting.

These charts visualize experimental results and are not a guarantee of future performance. Interpret them together with the metrics, transaction costs, spread, slippage, and the conditions of the data used.

## Features

- Feature engineering on OHLCV data, including the ATR indicator and historical features.
- A Gymnasium-based trading environment with automatic stop-loss and take-profit.
- PPO training with Stable-Baselines3 and observation normalization using `VecNormalize`.
- Evaluation on training, validation, and test data split chronologically.
- XAUUSD order execution through MetaTrader 5 with an ATR-multiplier-based stop-loss (`0.5`, `0.75`, or `1.0`) and an R-target-based take-profit (`0.5`, `0.75`, `1.0`, or `1.5`). Values are calculated when a new position is opened using the latest ATR.
- Automatic reversal: if the model predicts a direction opposite to the active position, the old position is closed immediately and a new position is opened with the direction and SL/TP from the latest prediction, without waiting for the old position's SL or TP to be hit.
- A FastAPI backend for position status and the history of closed positions.
- A React/Vite dashboard that refreshes data periodically.

### Position and Reversal Mechanism

The model produces three action components: direction (`0` = flat, `1` = long, `2` = short), the SL index, and the TP index. For a new position, the SL distance is calculated as `SL multiplier × ATR`, while the TP distance is calculated as `TP R multiplier × SL distance`. In the backend, the model is evaluated every 3 seconds. If the result is opposite to the active position, the system first attempts to close that position, then sends a new order with a volume of `0.5` lots.

SL/TP are protective and target levels set when the order is sent, not guarantees of execution prices. Spread, slippage, connectivity issues, broker rejection, or price changes may cause actual results to differ from the model's calculations.

## Brief Architecture

**Training flow**

```
Data CSV  →  Feature engineering  →  Trading environment  →  PPO training & backtesting
                                                                          ↓
                                                             Model (PPO + VecNormalize)
```

**Trading flow (runtime)**

```
Model  ─────────────────────────────────────────────────────────┐
                                                                 ↓
MetaTrader 5 (data harga XAUUSD)  →  FastAPI backend & trading loop  →  React dashboard
MetaTrader 5 (eksekusi order)     ←─────────────────────────────┘
```

The training flow uses CSV data to generate features, run simulations, train PPO, and produce a model. When trading mode is running, the backend loads the model, retrieves XAUUSD data from MetaTrader 5, sends orders based on the model's predictions, and provides position status and transaction history to the dashboard.

## Requirements

- Windows with Python `>= 3.12` for the backend.
- MetaTrader 5 desktop installed and accessible through the `MetaTrader5` Python package.
- Node.js and npm for the frontend.
- A broker **demo** account that supports the `XAUUSD` symbol for testing the integration. The symbol name may vary by broker.

## Installation

From the repository root directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r Requirements.txt
pip install -e .\main\backend
```

Instal dependensi frontend:

```powershell
cd main/frontend
npm install
cd ../..
```

Make sure the MetaTrader 5 terminal is open and a **demo account** is selected before running code that calls the MT5 API. Do not put account credentials in the source code or repository.

## Directory Structure

```text
.
├── main/
│   ├── backend/                # FastAPI, database, and trading runner
│   └── frontend/               # React/Vite dashboard
├── model/                      # PPO and VecNormalize models
├── notebooks/                  # Data exploration and model experiments
├── reinforcement_learning/    # Feature engineering, environment, and training
├── main_terminal_only.py       # Standalone terminal trading runner
├── Requirements.txt            # Main Python dependencies
└── README.md
```
