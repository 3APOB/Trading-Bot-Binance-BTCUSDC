# Zarov Binance Signal & Execution Bots (BTCUSDC)

This repository contains a two-process crypto trading system for Binance Spot, built around a simple and robust file-based communication mechanism using `signal.txt`.

The architecture separates market analysis from trade execution for clarity, safety, and modularity.

---

## Overview

The system is composed of two independent Python bots:

1) Signal Generator Bot (15m timeframe)  
2) Execution / Trader Bot (1m timeframe)

They communicate through a shared file called `signal.txt`.

Both bots send detailed logs and dashboards to Telegram.

---

## 1. Signal Generator Bot (15m)

Purpose:  
Analyze the market and generate high-quality BUY / SELL signals.

Main features:
- Fetches OHLC candles from Binance (15-minute timeframe)
- Stores data in a local SQLite database
- Computes advanced indicators:
  - RSI (14)
  - TEMA 20 / TEMA 50
  - Slope, speed, acceleration (based on TEMA20)
  - Local and global extrema
- Anti-false-signal protection:
  - Hysteresis filter (MIN_CROSS_PCT) to ignore micro-crosses
  - Confirmation filter (CONFIRM_BARS) requiring multiple candles
  - One signal maximum per candle (anti-spam)
- Emits signals only on confirmed regime changes:
  - SHORT → LONG  → BUY
  - LONG → SHORT  → SELL
- Writes signals atomically into `signal.txt`
- Sends confirmations and diagnostics to Telegram

Output:
- signal.txt containing either BUY, SELL, or empty

---

## 2. Execution / Trader Bot (1m)

Purpose:  
Execute trades based on signals and manage positions dynamically.

Main features:
- Fetches 1-minute close prices from Binance
- Stores candles in a local SQLite database
- Reads `signal.txt`
- Clears the file ONLY if a valid signal (BUY or SELL) is detected
- Simulated wallet management:
  - USDC balance
  - BTC balance
  - Trading fee
  - Fixed trade size (BTC)
- Dynamic position capacity calculation based on wallet and price
- Take-profit cycling strategy:
  - BUY → SELL when price reaches profit threshold
  - SELL → BUY when price retraces below threshold
- Continuous Telegram dashboard:
  - Trades
  - Take-profits
  - Wallet status
  - Total equity

Important note:  
This bot simulates trades internally. No real orders are sent unless you extend it.

---

## File Communication Logic

signal.txt is the bridge between the two bots.

Rules:
- Signal bot writes: BUY or SELL
- Execution bot reads the file
- If content is exactly BUY or SELL:
  - The signal is processed
  - The file is immediately cleared
- Any invalid content is ignored and kept for debugging

This guarantees:
- No duplicated trades
- No missed signals
- Simple and transparent synchronization

---

## Suggested Project Structure

.
├── trader_1m.py        Execution bot (1m)
├── signals_15m.py      Signal generator (15m)
├── signal.txt          Shared signal file (auto-created)
├── db_1m.db            SQLite database (1m candles)
├── db_15m.db           SQLite database (15m candles)
├── requirements.txt
└── README.md

---

## Requirements

- Python 3.10 or higher
- Binance account and API key
- Telegram bot token and chat ID

Python dependencies:
- python-binance
- pandas
- numpy
- python-telegram-bot
- sqlite3 (built-in)

---

## Installation

Create a virtual environment and install dependencies:

python -m venv .venv
activate the environment
pip install python-binance pandas numpy python-telegram-bot

---

## Configuration & Security

IMPORTANT:  
Never publish API keys or Telegram tokens in source code.

Use environment variables or a .env file instead.

Recommended variables:
- BINANCE_API_KEY
- BINANCE_API_SECRET
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID

---

## How to Run

Run the two bots in separate terminals or processes.

Terminal 1 – Signal Generator (15m):
python signals_15m.py

Terminal 2 – Execution Bot (1m):
python trader_1m.py

Both bots must run in the same directory to share signal.txt.

---

## Key Parameters

Signal Generator:
- MIN_CROSS_PCT: minimum relative distance between TEMA20 and TEMA50
- CONFIRM_BARS: number of candles required to confirm a regime change

Execution Bot:
- wallet_usdc: initial USDC balance
- wallet_btc: initial BTC balance
- fee_trade: trading fee
- qtt_trade_btc: trade size in BTC
- seuil: take-profit threshold

---

## Safety Notes

- This project is for educational and research purposes only
- No real trading risk management is enforced by default
- File-based signaling is simple but not suitable for high-frequency trading
- Do not run multiple instances using the same signal.txt

---

## Future Improvements

- Replace signal.txt with Redis, ZeroMQ, or WebSocket
- Add real Binance order execution
- Add risk management (max drawdown, daily stop)
- Add performance analytics and trade history

---

## Disclaimer

This software is provided as-is, without any warranty.
Cryptocurrency trading involves significant financial risk.
You are fully responsible for how you use this code.
