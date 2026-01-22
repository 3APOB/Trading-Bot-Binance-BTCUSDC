from binance.client import Client
import sqlite3, os, asyncio
import pandas as pd
from datetime import datetime, timedelta
from telegram import Bot

API_KEY = ""
API_SECRET = ""
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

SYMBOL, INTERVAL, DB_NAME, MAX_CANDLES = "BTCUSDC", "1m", "db_1m.db", 1000
BASE = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE, DB_NAME)
signal_path = os.path.join(BASE, "signal.txt")

client = Client(API_KEY, API_SECRET)
bot = Bot(TELEGRAM_TOKEN)

wallet_usdc, wallet_btc = 100.0, 0.0
fee_trade, qtt_trade_btc, seuil = 0.001, 0.0001, 0.0022

# Positions ouvertes (prix d'entrée)
last_buy_price, last_sell_price = [], []

# Slots restants (open possibles)
open_buy, open_sell, price = 0, 0, 0.0

# LOCK au démarrage (max fixes)
max_open_buy, max_open_sell = None, None


async def tg(msg):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        print(f"❌ Erreur Telegram: {e}")


def db_exec(sql, params=(), fetch=False):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql, params)
    out = cur.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return out


def get_last_price():
    r = db_exec("SELECT close FROM tc ORDER BY time DESC LIMIT 1", fetch=True)
    return float(r[0][0]) if r else None


def calcul_open_lock_once(p):
    """
    Calcule UNE SEULE FOIS au démarrage:
    - max_open_buy / max_open_sell : capacités verrouillées
    - open_buy / open_sell : slots restants initialement = max
    """
    global open_buy, open_sell, max_open_buy, max_open_sell

    if not p or p <= 0:
        max_open_buy, max_open_sell = 0, 0
        open_buy, open_sell = 0, 0
        return open_buy, open_sell

    # Déjà lock -> ne rien toucher
    if max_open_buy is not None and max_open_sell is not None:
        return open_buy, open_sell

    max_open_buy = max(0, int(wallet_usdc // (qtt_trade_btc * p * (1 + fee_trade))))
    max_open_sell = max(0, int(wallet_btc  // qtt_trade_btc))

    open_buy = max_open_buy
    open_sell = max_open_sell
    return open_buy, open_sell


async def wait_next_candle():
    now = datetime.now()
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    wait = (nxt - now).total_seconds()
    await tg(f"⏳ Attente {int(wait)}s jusqu'à {nxt:%H:%M:%S}")
    await asyncio.sleep(wait + 5)


async def read_signal():
    try:
        if not os.path.exists(signal_path):
            open(signal_path, "w").close()
            return None
        content = open(signal_path, "r").read().strip()
        if content:
            await tg(f"📄 Signal lu: '{content}'")
        sig = content.upper()
        if sig in ("BUY", "SELL"):
            open(signal_path, "w").close()
            await tg(f"✅ Signal '{sig}' traité et effacé")
            return sig
        if content:
            await tg(f"⚠️ Signal invalide ignoré: '{content}'")
        return None
    except Exception as e:
        await tg(f"❌ Erreur signal: {e}")
        return None


async def init_db():
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            await tg("♻️ DB existante supprimée")
        db_exec("CREATE TABLE IF NOT EXISTS tc (time INTEGER PRIMARY KEY, close REAL)")
        await tg("✅ DB initialisée")
    except Exception as e:
        await tg(f"❌ Erreur DB: {e}")


async def load_history():
    try:
        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=MAX_CANDLES)
        await tg(f"📥 {len(klines)} bougies reçues")
        conn = sqlite3.connect(db_path)
        conn.executemany(
            "INSERT OR IGNORE INTO tc VALUES (?, ?)",
            [(float(k[0]), float(k[4])) for k in klines]
        )
        conn.commit()
        df = pd.read_sql("SELECT * FROM tc ORDER BY time", conn)
        conn.close()
        last = df.iloc[-1]
        ts = datetime.fromtimestamp(last["time"] / 1000)
        await tg(f"✅ {len(df)} bougies\n🕐 {ts:%Y-%m-%d %H:%M}\n💰 {last['close']:.2f} USDC")

        # LOCK OPEN LIMITS ICI (1 seule fois)
        calcul_open_lock_once(float(last["close"]))
        await tg(
            f"🔒 <b>OPEN LOCK</b>\n"
            f"Max BUY: {max_open_buy} | Max SELL: {max_open_sell}\n"
            f"Slots BUY: {open_buy} | Slots SELL: {open_sell}"
        )
    except Exception as e:
        await tg(f"❌ Erreur load: {e}")


async def add_candle():
    try:
        k = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1)[0]
        db_exec("INSERT OR IGNORE INTO tc VALUES (?, ?)", (float(k[0]), float(k[4])))
        ts = datetime.fromtimestamp(float(k[0]) / 1000)
        await tg(f"🆕 Bougie\n🕐 {ts:%H:%M}\n💰 {float(k[4]):.2f}")
    except Exception as e:
        await tg(f"❌ Erreur add: {e}")


async def trade():
    global wallet_usdc, wallet_btc, price, open_buy, open_sell
    try:
        sig = await read_signal()
        price = get_last_price()
        if not price:
            return

        # sécurité: si pas lock (au cas où)
        calcul_open_lock_once(price)

        # =========================
        # OUVERTURE LONG (BUY)
        # =========================
        if sig == "BUY":
            cost = qtt_trade_btc * price * (1 + fee_trade)
            if wallet_usdc >= cost and open_buy > 0:
                last_buy_price.append(price)
                wallet_btc += qtt_trade_btc
                wallet_usdc -= cost
                open_buy -= 1  # consomme 1 slot

                await tg(
                    f"✅ <b>ACHAT</b>\n💰 {price:.2f} USDC\n₿ {qtt_trade_btc:.6f} BTC\n"
                    f"💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n"
                    f"📌 Longs ouverts: {len(last_buy_price)}\n"
                    f"🔒 Slots BUY restants: {open_buy}/{max_open_buy}"
                )
            else:
                await tg(
                    f"⚠️ BUY refusé\n"
                    f"💵 Requis: {cost:.2f}\n💵 Dispo: {wallet_usdc:.2f}\n"
                    f"🔒 Slots BUY: {open_buy}/{max_open_buy}"
                )

        # =========================
        # TP LONG (fermeture -> vente)
        # =========================
        for i in range(len(last_buy_price) - 1, -1, -1):
            bp = last_buy_price[i]
            if bp * (1 + seuil) <= price and wallet_btc >= qtt_trade_btc:
                last_buy_price.pop(i)

                wallet_btc -= qtt_trade_btc
                wallet_usdc += qtt_trade_btc * price * (1 - fee_trade)

                # libère 1 slot BUY (sans dépasser max)
                open_buy = min(max_open_buy, open_buy + 1)

                prf = (price - bp) / bp * 100
                await tg(
                    f"✅ <b>TP VENTE (CLOSE LONG)</b>\n"
                    f"💰 Achat: {bp:.2f}\n💰 Vente: {price:.2f}\n📈 +{prf:.2f}%\n"
                    f"💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n"
                    f"🔒 Slots BUY: {open_buy}/{max_open_buy}"
                )

        # =========================
        # OUVERTURE SHORT (SELL)
        # =========================
        if sig == "SELL":
            if wallet_btc >= qtt_trade_btc and open_sell > 0:
                last_sell_price.append(price)
                wallet_btc -= qtt_trade_btc
                wallet_usdc += qtt_trade_btc * price * (1 - fee_trade)
                open_sell -= 1  # consomme 1 slot

                await tg(
                    f"✅ <b>VENTE (OPEN SHORT)</b>\n💰 {price:.2f} USDC\n₿ {qtt_trade_btc:.6f} BTC\n"
                    f"💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n"
                    f"📌 Shorts ouverts: {len(last_sell_price)}\n"
                    f"🔒 Slots SELL restants: {open_sell}/{max_open_sell}"
                )
            else:
                await tg(
                    f"⚠️ SELL refusé\n"
                    f"₿ Requis: {qtt_trade_btc:.6f}\n₿ Dispo: {wallet_btc:.6f}\n"
                    f"🔒 Slots SELL: {open_sell}/{max_open_sell}"
                )

        # =========================
        # TP SHORT (fermeture -> rachat)
        # =========================
        for i in range(len(last_sell_price) - 1, -1, -1):
            sp = last_sell_price[i]
            cost = qtt_trade_btc * price * (1 + fee_trade)

            # Short gagnant si prix a baissé d'au moins seuil
            if sp >= price * (1 + seuil) and wallet_usdc >= cost:
                last_sell_price.pop(i)

                wallet_btc += qtt_trade_btc
                wallet_usdc -= cost

                # libère 1 slot SELL (sans dépasser max)
                open_sell = min(max_open_sell, open_sell + 1)

                prf = (sp - price) / sp * 100
                await tg(
                    f"✅ <b>TP RACHAT (CLOSE SHORT)</b>\n"
                    f"💰 Vente: {sp:.2f}\n💰 Rachat: {price:.2f}\n📈 +{prf:.2f}%\n"
                    f"💵 USDC: {wallet_usdc:.2f}\n₿ BTC: {wallet_btc:.6f}\n"
                    f"🔒 Slots SELL: {open_sell}/{max_open_sell}"
                )

    except Exception as e:
        await tg(f"❌ Erreur trade: {e}")


async def display_status():
    p = get_last_price()
    if not p:
        return
    tot = wallet_usdc + (wallet_btc * p * (1 - fee_trade))
    await tg(
        f"📊 <b>STATUT</b>\n"
        f"💰 {p:.2f} USDC\n"
        f"💵 {wallet_usdc:.2f}\n"
        f"₿ {wallet_btc:.6f}\n"
        f"💎 Total: {tot:.2f}\n"
        f"🔒 BUY slots: {open_buy}/{max_open_buy} | SELL slots: {open_sell}/{max_open_sell}\n"
        f"📌 Longs: {len(last_buy_price)} | Shorts: {len(last_sell_price)}"
    )


async def main():
    try:
        await init_db()
        await load_history()
        await tg(
            f"🤖 <b>BOT DÉMARRÉ</b>\n📊 {SYMBOL}\n⏱️ {INTERVAL}\n📈 Seuil: {seuil*100}%\n✅ En attente..."
        )
        while True:
            await wait_next_candle()
            await add_candle()
            await trade()
            await display_status()
    except KeyboardInterrupt:
        await tg("⏹️ Bot arrêté")
    except Exception as e:
        await tg(f"❌ Erreur: {e}")


if __name__ == "__main__":
    asyncio.run(main())
