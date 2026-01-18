# =========================
# 🔐 LIBRAIRIE
# =========================
from binance.client import Client
import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from telegram import Bot
import asyncio

# =========================
# 🔐 CONFIG
# =========================
API_KEY = ""
API_SECRET = ""

# 🔔 Configuration Telegram
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

SYMBOL = "BTCUSDC"
INTERVAL = "15m"
DB_NAME = "db_15m.db"
MAX_CANDLES = 1000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, DB_NAME)

# ✅ signal.txt dans le même dossier que le script
SIGNAL_FILE = os.path.join(BASE_DIR, "signal.txt")

client = Client(API_KEY, API_SECRET)

# =========================
# ✅ ANTI-FAUX SIGNAUX (tuning)
# =========================
# Si la différence relative TEMA20/TEMA50 est trop faible => NEUTRAL (évite les micro-croisements)
MIN_CROSS_PCT = 0.00015   # 0.015% (ajuste: 0.00010 à 0.00050 selon ton bruit)
# Confirmation: il faut 2 bougies d’affilée dans le nouveau régime
CONFIRM_BARS = 2

# ✅ mémoire d’état
last_regime = None          # "LONG" | "SHORT" | "NEUTRAL" | None
last_signal_time_ms = None  # anti spam par bougie


async def send_telegram(message: str):
    try:
        bot = Bot(TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"❌ Telegram error: {e}")


# =====================
# 🔧 CALCUL DES INDICATEURS
# =====================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_tema(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def calculate_slope(series, period=20):
    slope = series.rolling(window=period).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == period else np.nan,
        raw=True
    )
    return slope


def calculate_speed(series, period=20):
    return series.diff(period) / period


def calculate_acceleration(series, period=20):
    speed = series.diff(period) / period
    return speed.diff(period) / period


def calculate_local_extrema(series, period=20):
    local_max = series.rolling(window=period, center=True).max()
    local_min = series.rolling(window=period, center=True).min()
    return local_max, local_min


def calculate_global_extrema(series, period=20):
    global_max = series.rolling(window=period, min_periods=1).max()
    global_min = series.rolling(window=period, min_periods=1).min()
    return global_max, global_min


def calculate_all_indicators(df):
    df = df.copy()
    nb_candles = len(df)

    if nb_candles >= 14:
        df['rsi'] = calculate_rsi(df['close'], period=14)
    else:
        df['rsi'] = None

    if nb_candles >= 60:
        df['tema20'] = calculate_tema(df['close'], period=20)
        df['slope20'] = calculate_slope(df['tema20'], period=20)
        df['speed20'] = calculate_speed(df['tema20'], period=20)
        df['acceleration20'] = calculate_acceleration(df['tema20'], period=20)
        df['local_max20'], df['local_min20'] = calculate_local_extrema(df['tema20'], period=20)
        df['global_max20'], df['global_min20'] = calculate_global_extrema(df['tema20'], period=20)
    else:
        for col in ['tema20', 'slope20', 'speed20', 'acceleration20',
                    'local_max20', 'local_min20', 'global_max20', 'global_min20']:
            df[col] = None

    if nb_candles >= 150:
        df['tema50'] = calculate_tema(df['close'], period=50)
    else:
        df['tema50'] = None

    df.loc[df.index < 14, 'rsi'] = None
    df.loc[df.index < 60, ['tema20', 'slope20', 'speed20', 'acceleration20',
                           'local_max20', 'local_min20', 'global_max20', 'global_min20']] = None
    df.loc[df.index < 150, 'tema50'] = None

    return df


# =========================
# 🧠 SIGNAL LOGIC
# =========================

def _regime_from_tema(tema20, tema50, close=None) -> str:
    if tema20 is None or tema50 is None:
        return "NEUTRAL"
    if pd.isna(tema20) or pd.isna(tema50):
        return "NEUTRAL"

    t20 = float(tema20)
    t50 = float(tema50)

    # Hystérésis: si l'écart est trop petit -> NEUTRAL
    if close is not None and close > 0:
        diff_pct = abs(t20 - t50) / float(close)
        if diff_pct < MIN_CROSS_PCT:
            return "NEUTRAL"

    if t20 > t50:
        return "LONG"
    if t20 < t50:
        return "SHORT"
    return "NEUTRAL"


def write_signal(signal: str):
    """Écrit le signal dans signal.txt (atomique)."""
    try:
        tmp = SIGNAL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(signal.strip())
        os.replace(tmp, SIGNAL_FILE)
    except Exception as e:
        print(f"❌ Erreur write_signal: {e}")


async def detect_and_emit_signal():
    """
    Anti faux signaux:
    - calcule le régime sur plusieurs bougies
    - confirmation: nouveau régime doit être présent sur CONFIRM_BARS bougies d'affilée
    - hystérésis: petit écart => NEUTRAL
    """
    global last_regime, last_signal_time_ms

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT time, close, tema20, tema50
            FROM ohlc
            ORDER BY time DESC
            LIMIT {max(3, CONFIRM_BARS + 1)}
        """)
        rows = cur.fetchall()

        if len(rows) < max(2, CONFIRM_BARS + 1):
            write_signal("")
            return

        # rows[0] = dernière bougie
        last_time = int(rows[0][0])
        last_close = float(rows[0][1]) if rows[0][1] is not None else None

        # Anti-spam: si on a déjà émis un signal sur cette bougie, on ne refait rien
        if last_signal_time_ms == last_time:
            write_signal("")
            return

        # Régimes récents
        regimes = []
        for r in rows[:CONFIRM_BARS]:
            t, close, t20, t50 = r
            regimes.append(_regime_from_tema(t20, t50, close=float(close) if close else None))

        # Régime courant confirmé (les N dernières bougies doivent être identiques et non NEUTRAL)
        curr_regime = regimes[0]
        if curr_regime == "NEUTRAL":
            write_signal("")
            last_regime = "NEUTRAL"
            return

        if not all(x == curr_regime for x in regimes):
            # Pas confirmé
            write_signal("")
            return

        # Initialise mémoire au premier passage
        if last_regime is None:
            last_regime = curr_regime
            write_signal("")
            return

        # Aucun changement -> rien
        if curr_regime == last_regime:
            write_signal("")
            return

        # On ne déclenche que sur LONG <-> SHORT (NEUTRAL déjà filtré)
        signal_to_write = ""
        if last_regime == "SHORT" and curr_regime == "LONG":
            signal_to_write = "BUY"
        elif last_regime == "LONG" and curr_regime == "SHORT":
            signal_to_write = "SELL"

        write_signal(signal_to_write)

        if signal_to_write:
            t = datetime.fromtimestamp(last_time / 1000)
            msg = (
                f"🚨 <b>SIGNAL CONFIRMÉ</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🕒 Bougie: {t.strftime('%Y-%m-%d %H:%M')}\n"
                f"🔁 {last_regime} ➜ {curr_regime} (x{CONFIRM_BARS})\n"
                f"📌 <b>{signal_to_write}</b>\n"
                f"💰 Close: {last_close:.2f}\n"
                f"🧯 Filtre: MIN_CROSS_PCT={MIN_CROSS_PCT*100:.4f}%"
            )
            print(msg)
            await send_telegram(msg)
            last_signal_time_ms = last_time

        last_regime = curr_regime

    finally:
        conn.close()


# =========================
# 🗄️ DB
# =========================

async def init_db():
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            message = "♻️ Base de données existante détectée et supprimée."
        else:
            message = "🆕 Aucune base existante — création d'une nouvelle base de données..."
        print(message)
        await send_telegram(message)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlc (
                time INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                rsi REAL,
                tema20 REAL,
                tema50 REAL,
                slope20 REAL,
                speed20 REAL,
                acceleration20 REAL,
                local_max20 REAL,
                local_min20 REAL,
                global_max20 REAL,
                global_min20 REAL
            )
        """)

        conn.commit()
        conn.close()

        message = (
            "✅ Base de données initialisée avec succès\n"
            "📊 Table : OHLC + indicateurs techniques"
        )
        print(message)
        await send_telegram(message)

    except sqlite3.Error as e:
        message = f"❌ ERREUR BASE DE DONNÉES\n🧱 SQLite : {e}"
        print(message)
        await send_telegram(message)


async def load_history():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        message = f"📥 Récupération de {MAX_CANDLES} bougies historiques..."
        print(message)
        await send_telegram(message)

        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=MAX_CANDLES)

        message = f"📊 {len(klines)} bougies reçues de Binance"
        print(message)
        await send_telegram(message)

        for kline in klines:
            cursor.execute("""
                INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (int(kline[0]), float(kline[1]), float(kline[2]),
                  float(kline[3]), float(kline[4]), float(kline[5])))

        conn.commit()

        message = "🧮 Calcul des indicateurs techniques en cours..."
        print(message)
        await send_telegram(message)

        df = pd.read_sql("SELECT * FROM ohlc ORDER BY time", conn)
        df = calculate_all_indicators(df)

        for _, row in df.iterrows():
            cursor.execute("""
                UPDATE ohlc
                SET rsi = ?, tema20 = ?, tema50 = ?, slope20 = ?, speed20 = ?,
                    acceleration20 = ?, local_max20 = ?, local_min20 = ?,
                    global_max20 = ?, global_min20 = ?
                WHERE time = ?
            """, (
                None if pd.isna(row['rsi']) else float(row['rsi']),
                None if pd.isna(row['tema20']) else float(row['tema20']),
                None if pd.isna(row['tema50']) else float(row['tema50']),
                None if pd.isna(row['slope20']) else float(row['slope20']),
                None if pd.isna(row['speed20']) else float(row['speed20']),
                None if pd.isna(row['acceleration20']) else float(row['acceleration20']),
                None if pd.isna(row['local_max20']) else float(row['local_max20']),
                None if pd.isna(row['local_min20']) else float(row['local_min20']),
                None if pd.isna(row['global_max20']) else float(row['global_max20']),
                None if pd.isna(row['global_min20']) else float(row['global_min20']),
                int(row['time'])
            ))

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM ohlc")
        total = cursor.fetchone()[0]

        last_row = df.iloc[-1]
        timestamp = datetime.fromtimestamp(last_row['time'] / 1000)

        rsi_val = f"{last_row['rsi']:.2f}" if pd.notna(last_row['rsi']) else 'N/A'
        tema20_val = f"{last_row['tema20']:.2f}" if pd.notna(last_row['tema20']) else 'N/A'
        tema50_val = f"{last_row['tema50']:.2f}" if pd.notna(last_row['tema50']) else 'N/A'

        message = (
            f"✅ <b>{total} bougies chargées avec succès</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 Dernière bougie : {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"💰 Close : {float(last_row['close']):.2f} USDC\n"
            f"📊 RSI : {rsi_val}\n"
            f"📈 TEMA20 : {tema20_val}\n"
            f"📉 TEMA50 : {tema50_val}"
        )
        print(message)
        await send_telegram(message)

        conn.close()

        # ✅ Initialise last_regime proprement sans générer de signal
        await detect_and_emit_signal()

    except Exception as e:
        message = f"❌ Erreur lors du chargement : {e}"
        print(message)
        await send_telegram(message)


async def add_candle():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1)

        for kline in klines:
            cursor.execute("""
                INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (int(kline[0]), float(kline[1]), float(kline[2]),
                  float(kline[3]), float(kline[4]), float(kline[5])))

        conn.commit()

        cursor.execute("SELECT * FROM ohlc ORDER BY time DESC LIMIT 200")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows[::-1], columns=columns)

        df = calculate_all_indicators(df)

        last_row = df.iloc[-1]
        cursor.execute("""
            UPDATE ohlc
            SET rsi = ?, tema20 = ?, tema50 = ?, slope20 = ?, speed20 = ?,
                acceleration20 = ?, local_max20 = ?, local_min20 = ?,
                global_max20 = ?, global_min20 = ?
            WHERE time = ?
        """, (
            None if pd.isna(last_row['rsi']) else float(last_row['rsi']),
            None if pd.isna(last_row['tema20']) else float(last_row['tema20']),
            None if pd.isna(last_row['tema50']) else float(last_row['tema50']),
            None if pd.isna(last_row['slope20']) else float(last_row['slope20']),
            None if pd.isna(last_row['speed20']) else float(last_row['speed20']),
            None if pd.isna(last_row['acceleration20']) else float(last_row['acceleration20']),
            None if pd.isna(last_row['local_max20']) else float(last_row['local_max20']),
            None if pd.isna(last_row['local_min20']) else float(last_row['local_min20']),
            None if pd.isna(last_row['global_max20']) else float(last_row['global_max20']),
            None if pd.isna(last_row['global_min20']) else float(last_row['global_min20']),
            int(last_row['time'])
        ))

        conn.commit()

        timestamp = datetime.fromtimestamp(last_row['time'] / 1000)
        message = (
            f"🆕 <b>Nouvelle bougie ajoutée</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 Timestamp : {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"💰 Close : {float(last_row['close']):.2f} USDC"
        )
        print(message)
        await send_telegram(message)

        conn.close()

        # ✅ Détection signal confirmée
        await detect_and_emit_signal()

    except Exception as e:
        message = f"❌ Erreur lors de l'ajout : {e}"
        print(message)
        await send_telegram(message)


def interval_to_seconds(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60


async def wait_next_candle():
    sec = interval_to_seconds(INTERVAL)
    now = time.time()
    next_ts = ((int(now) // sec) + 1) * sec
    wait_seconds = max(0, next_ts - now) + 2

    next_dt = datetime.fromtimestamp(next_ts)

    message = (
        f"⏰ Heure actuelle: {datetime.now().strftime('%H:%M:%S')}\n"
        f"⏳ Prochaine bougie à: {next_dt.strftime('%H:%M:%S')}\n"
        f"⏱️ Attente de {int(wait_seconds)} secondes..."
    )
    print(message)
    await send_telegram(message)

    await asyncio.sleep(wait_seconds)


async def show_infos():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT time, close, tema20, tema50 FROM ohlc ORDER BY time DESC LIMIT 1")
        row = cursor.fetchone()

        if row:
            timestamp = datetime.fromtimestamp(row[0] / 1000)
            close = float(row[1])
            tema20 = row[2]
            tema50 = row[3]

            regime = _regime_from_tema(tema20, tema50, close=close)
            if regime == "LONG":
                signal = "LONG 📈"
            elif regime == "SHORT":
                signal = "SHORT 📉"
            else:
                signal = "NEUTRE ⚪"

            message = (
                f"⚡ Régime : {signal}\n"
                f"🕐 {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                f"💰 Close : {close:.2f}\n"
                f"🧪 MIN_CROSS_PCT : {MIN_CROSS_PCT*100:.4f}%\n"
                f"✅ Confirmation : x{CONFIRM_BARS}\n"
                f"📄 signal.txt : {SIGNAL_FILE}"
            )
            print(message)
            await send_telegram(message)
        else:
            message = "⚠️ Pas de bougie trouvée dans la base de données."
            print(message)
            await send_telegram(message)

        conn.close()
    except Exception as e:
        print(f"❌ Erreur show_infos : {e}")
        await send_telegram(f"❌ Erreur show_infos : {e}")


async def bot_async():
    write_signal("")

    await init_db()

    message = (
        "🤖 BOT SIGNALS LANCÉ\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚙️ Initialisation..."
    )
    print(message)
    await send_telegram(message)

    await load_history()

    message = (
        "🚀 <b>Bot opérationnel</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Symbole    : {SYMBOL}\n"
        f"⏱️ Timeframe  : {INTERVAL}\n"
        "📊 Stratégie : TEMA 20 / TEMA 50\n"
        f"🧯 Filtre: MIN_CROSS_PCT={MIN_CROSS_PCT*100:.4f}%\n"
        f"✅ Confirmation: x{CONFIRM_BARS}\n"
        f"📝 Signal file : {SIGNAL_FILE}"
    )
    print(message)
    await send_telegram(message)

    while True:
        await show_infos()
        await wait_next_candle()
        await add_candle()


def bot():
    asyncio.run(bot_async())


# =========================
if __name__ == "__main__":
    try:
        bot()
    except KeyboardInterrupt:
        stop_message = (
            "🛑 <b>Arrêt manuel du bot</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "📴 Exécution interrompue par l'utilisateur"
        )
        print("\n🛑 Bot arrêté proprement")
        asyncio.run(send_telegram(stop_message))
