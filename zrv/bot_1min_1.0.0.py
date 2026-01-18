from binance.client import Client
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from telegram import Bot
import asyncio

# CONFIG
API_KEY = ""
API_SECRET = ""
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
SYMBOL, INTERVAL, DB_NAME, MAX_CANDLES = "BTCUSDC", "1m", "db_1m.db", 1000

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
signal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal.txt")
client = Client(API_KEY, API_SECRET)

# WALLET
wallet_usdc, wallet_btc, fee_trade, qtt_trade_btc, seuil = 100.0, 0.0, 0.001, 0.0001, 0.0022
last_buy_price, last_sell_price, open_buy, open_sell, price = [], [], 0, 0, 0.0

async def send_telegram(msg):
    try:
        await Bot(TELEGRAM_TOKEN).send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')
        print("✅ Telegram envoyé")
    except Exception as e:
        print(f"❌ Erreur Telegram: {e}")

async def init_db():
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            await send_telegram("♻️ DB existante supprimée")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS tc (time INTEGER PRIMARY KEY, close REAL)")
        conn.commit()
        conn.close()
        await send_telegram("✅ DB initialisée")
    except Exception as e:
        await send_telegram(f"❌ Erreur DB: {e}")

async def load_history():
    try:
        conn = sqlite3.connect(db_path)
        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=MAX_CANDLES)
        await send_telegram(f"📥 {len(klines)} bougies reçues")
        for k in klines:
            conn.execute("INSERT OR IGNORE INTO tc VALUES (?, ?)", (float(k[0]), float(k[4])))
        conn.commit()
        df = pd.read_sql("SELECT * FROM tc ORDER BY time", conn)
        last = df.iloc[-1]
        ts = datetime.fromtimestamp(last['time'] / 1000)
        await send_telegram(f"✅ {len(df)} bougies\n🕐 {ts:%Y-%m-%d %H:%M}\n💰 {last['close']:.2f} USDC")
        calcul_open(last['close'])
        conn.close()
    except Exception as e:
        await send_telegram(f"❌ Erreur load: {e}")

async def add_candle():
    try:
        conn = sqlite3.connect(db_path)
        k = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1)[0]
        conn.execute("INSERT OR IGNORE INTO tc VALUES (?, ?)", (float(k[0]), float(k[4])))
        conn.commit()
        df = pd.read_sql("SELECT * FROM tc ORDER BY time DESC LIMIT 1", conn)
        last = df.iloc[0]
        ts = datetime.fromtimestamp(last['time'] / 1000)
        await send_telegram(f"🆕 Bougie\n🕐 {ts:%H:%M}\n💰 {last['close']:.2f}")
        conn.close()
    except Exception as e:
        await send_telegram(f"❌ Erreur add: {e}")

async def wait_next_candle():
    now = datetime.now()
    next_min = ((now.minute // 1) + 1) * 1
    next_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)) if next_min >= 60 else now.replace(minute=next_min, second=0, microsecond=0)
    wait = (next_time - now).total_seconds()
    await send_telegram(f"⏳ Attente {int(wait)}s jusqu'à {next_time:%H:%M:%S}")
    await asyncio.sleep(wait + 5)

async def read_signal():
    try:
        if not os.path.exists(signal_path):
            open(signal_path, 'w').close()
            return None
        
        with open(signal_path, 'r') as f:
            content = f.read().strip()
        
        # Debug : afficher ce qui a été lu
        if content:
            await send_telegram(f"📄 Signal lu: '{content}'")
        
        sig = content.upper()
        
        # Ne vider le fichier QUE si un signal valide est détecté
        if sig in ["BUY", "SELL"]:
            open(signal_path, 'w').close()
            await send_telegram(f"✅ Signal '{sig}' traité et effacé")
            return sig
        elif content:  # Si quelque chose est écrit mais pas valide
            await send_telegram(f"⚠️ Signal invalide ignoré: '{content}'")
        
        return None
    except Exception as e:
        await send_telegram(f"❌ Erreur signal: {e}")
        return None

def calcul_open(price):
    global open_buy, open_sell

    # sécurité: éviter division / valeurs invalides
    if price is None or price <= 0:
        open_buy, open_sell = 0, 0
        return open_buy, open_sell

    buy_cap = int(wallet_usdc // (qtt_trade_btc * price * (1 + fee_trade)))
    sell_cap = int(wallet_btc // qtt_trade_btc)

    # ICI: pas de min() sur un int -> c'est ça qui plantait
    open_buy = max(0, buy_cap)
    open_sell = max(0, sell_cap)


async def trade():
    global wallet_usdc, wallet_btc, last_buy_price, last_sell_price, price, open_buy, open_sell
    try:
        sig = await read_signal()
        
        # Debug : indiquer si aucun signal
        if sig is None:
            print("Aucun signal détecté")  # Log console uniquement
        
        conn = sqlite3.connect(db_path)
        res = conn.execute("SELECT close FROM tc ORDER BY time DESC LIMIT 1").fetchone()
        conn.close()
        if not res:
            return
        price = res[0]

        

        # ACHAT
        if sig == "BUY":
            if wallet_usdc > qtt_trade_btc * price * (1 + fee_trade) and open_buy > 0:
                last_buy_price.append(price)
                wallet_btc += qtt_trade_btc
                wallet_usdc -= qtt_trade_btc * price * (1 + fee_trade)
                open_buy -= 1
                await send_telegram(f"✅ <b>ACHAT</b>\n💰 {price:.2f} USDC\n₿ {qtt_trade_btc:.6f} BTC\n💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n📊 Pos BUY: {len(last_buy_price)}")
            else:
                await send_telegram(f"⚠️ Signal BUY reçu mais fonds insuffisants\n💵 Requis: {qtt_trade_btc * price * (1 + fee_trade):.2f}\n💵 Disponible: {wallet_usdc:.2f}")
        
        # TP ACHATS (A-V)
        for i in range(len(last_buy_price) - 1, -1, -1):
            if last_buy_price[i] * (1 + seuil) <= price and wallet_btc >= qtt_trade_btc:
                bp = last_buy_price.pop(i)
                last_sell_price.append(price)
                wallet_btc -= qtt_trade_btc
                wallet_usdc += qtt_trade_btc * price * (1 - fee_trade)
                prf = (price - bp) / bp * 100
                await send_telegram(f"✅ <b>TP VENTE</b>\n💰 Achat: {bp:.2f}\n💰 Vente: {price:.2f}\n📈 +{prf:.2f}%\n💵 {wallet_usdc:.2f} USDC")
        
        # VENTE
        if sig == "SELL":
            if wallet_btc >= qtt_trade_btc and open_sell > 0:
                last_sell_price.append(price)
                wallet_btc -= qtt_trade_btc
                wallet_usdc += qtt_trade_btc * price * (1 - fee_trade)
                open_sell -= 1
                await send_telegram(f"✅ <b>VENTE</b>\n💰 {price:.2f} USDC\n₿ {qtt_trade_btc:.6f} BTC\n💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n📊 Pos SELL: {len(last_sell_price)}")
            else:
                await send_telegram(f"⚠️ Signal SELL reçu mais BTC insuffisants\n₿ Requis: {qtt_trade_btc:.6f}\n₿ Disponible: {wallet_btc:.6f}")
        
        # TP VENTES (V-A)
        for i in range(len(last_sell_price) - 1, -1, -1):
            if last_sell_price[i] >= price * (1 + seuil) and wallet_usdc >= qtt_trade_btc * price * (1 + fee_trade):
                sp = last_sell_price.pop(i)
                last_buy_price.append(price)
                wallet_btc += qtt_trade_btc
                wallet_usdc -= qtt_trade_btc * price * (1 + fee_trade)
                prf = (sp - price) / sp * 100
                await send_telegram(f"✅ <b>TP RACHAT</b>\n💰 Vente: {sp:.2f}\n💰 Rachat: {price:.2f}\n📈 +{prf:.2f}%\n₿ {wallet_btc:.6f} BTC")
    except Exception as e:
        await send_telegram(f"❌ Erreur trade: {e}")

async def display_status():
    global open_sell, open_buy
    # on récupère d'abord le prix, puis on appelle calcul_open(price)
    conn = sqlite3.connect(db_path)
    res = conn.execute("SELECT close FROM tc ORDER BY time DESC LIMIT 1").fetchone()
    conn.close()

    if res:
        p = res[0]
        tot = wallet_usdc + (wallet_btc * p)
        await send_telegram(f"📊 <b>STATUT</b>\n💰 {p:.2f} USDC\n💵 {wallet_usdc:.2f}\n₿ {wallet_btc:.6f}\n💎 Total: {tot:.2f}\n📊 BUY: {open_buy} | SELL: {open_sell}")

async def main():
    try:
        await init_db()
        await load_history()
        await send_telegram(f"🤖 <b>BOT DÉMARRÉ</b>\n📊 {SYMBOL}\n⏱️ {INTERVAL}\n📈 Seuil: {seuil*100}%\n✅ En attente...")
        while True:
            await wait_next_candle()
            await add_candle()
            await trade()
            await display_status()
    except KeyboardInterrupt:
        await send_telegram("⏹️ Bot arrêté")
    except Exception as e:
        await send_telegram(f"❌ Erreur: {e}")

if __name__ == "__main__":
    asyncio.run(main())
