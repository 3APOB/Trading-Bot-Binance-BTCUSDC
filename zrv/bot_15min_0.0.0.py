from binance.client import Client
import sqlite3, pandas as pd, numpy as np, time, os, asyncio
from datetime import datetime
from telegram import Bot

API_KEY=""
API_SECRET=""
TELEGRAM_TOKEN=""
TELEGRAM_CHAT_ID=""
SYMBOL, INTERVAL, DB_NAME, MAX_CANDLES = "BTCUSDC", "15m", "db_15m.db", 1000

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
db_path=os.path.join(BASE_DIR, DB_NAME)
SIGNAL_FILE=os.path.join(BASE_DIR, "signal.txt")
client=Client(API_KEY, API_SECRET)

last_regime=None          # "LONG" | "SHORT" | None
last_signal_time_ms=None  # anti spam par bougie

async def send_telegram(message:str):
    try: await Bot(TELEGRAM_TOKEN).send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="HTML")
    except Exception as e: print(f"❌ Telegram error: {e}")

def calculate_rsi(series, period=14):
    delta=series.diff()
    gain=(delta.where(delta>0,0)).rolling(window=period).mean()
    loss=(-delta.where(delta<0,0)).rolling(window=period).mean()
    rs=gain/loss
    return 100-(100/(1+rs))

def calculate_tema(series, period):
    ema1=series.ewm(span=period, adjust=False).mean()
    ema2=ema1.ewm(span=period, adjust=False).mean()
    ema3=ema2.ewm(span=period, adjust=False).mean()
    return 3*ema1-3*ema2+ema3

def calculate_slope(series, period=20):
    return series.rolling(window=period).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==period else np.nan, raw=True
    )

def calculate_speed(series, period=20): return series.diff(period)/period
def calculate_acceleration(series, period=20):
    speed=series.diff(period)/period
    return speed.diff(period)/period

def calculate_local_extrema(series, period=20):
    return series.rolling(window=period, center=True).max(), series.rolling(window=period, center=True).min()

def calculate_global_extrema(series, period=20):
    return series.rolling(window=period, min_periods=1).max(), series.rolling(window=period, min_periods=1).min()

def calculate_all_indicators(df):
    df=df.copy(); n=len(df)
    df["rsi"]=calculate_rsi(df["close"],14) if n>=14 else None
    if n>=60:
        df["tema20"]=calculate_tema(df["close"],20)
        df["slope20"]=calculate_slope(df["tema20"],20)
        df["speed20"]=calculate_speed(df["tema20"],20)
        df["acceleration20"]=calculate_acceleration(df["tema20"],20)
        df["local_max20"], df["local_min20"]=calculate_local_extrema(df["tema20"],20)
        df["global_max20"], df["global_min20"]=calculate_global_extrema(df["tema20"],20)
    else:
        for c in ["tema20","slope20","speed20","acceleration20","local_max20","local_min20","global_max20","global_min20"]:
            df[c]=None
    df["tema50"]=calculate_tema(df["close"],50) if n>=150 else None
    df.loc[df.index<14,"rsi"]=None
    df.loc[df.index<60,["tema20","slope20","speed20","acceleration20","local_max20","local_min20","global_max20","global_min20"]]=None
    df.loc[df.index<150,"tema50"]=None
    return df

def _regime_from_tema(tema20, tema50)->str:
    if tema20 is None or tema50 is None or pd.isna(tema20) or pd.isna(tema50): return None
    return "LONG" if float(tema20)>float(tema50) else "SHORT"

def write_signal(signal:str):
    try:
        tmp=SIGNAL_FILE+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: f.write(signal.strip())
        os.replace(tmp, SIGNAL_FILE)
    except Exception as e: print(f"❌ Erreur write_signal: {e}")

async def detect_and_emit_signal():
    global last_regime, last_signal_time_ms
    conn=sqlite3.connect(db_path)
    try:
        cur=conn.cursor()
        cur.execute("SELECT time, close, tema20, tema50 FROM ohlc ORDER BY time DESC LIMIT 2")
        rows=cur.fetchall()
        if len(rows)<1: write_signal(""); return
        last_time=int(rows[0][0])
        last_close=float(rows[0][1]) if rows[0][1] is not None else None
        last_t20, last_t50 = rows[0][2], rows[0][3]
        if last_signal_time_ms==last_time: write_signal(""); return
        curr_regime=_regime_from_tema(last_t20, last_t50)
        if curr_regime is None: write_signal(""); last_signal_time_ms=last_time; return
        if last_regime is None:
            last_regime=curr_regime; write_signal(""); last_signal_time_ms=last_time; return
        if curr_regime==last_regime: write_signal(""); last_signal_time_ms=last_time; return
        signal_to_write="BUY" if (last_regime=="SHORT" and curr_regime=="LONG") else "SELL"
        write_signal(signal_to_write)
        t=datetime.fromtimestamp(last_time/1000)
        msg=(f"🚨 <b>SIGNAL</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
             f"🕒 Bougie: {t.strftime('%Y-%m-%d %H:%M')}\n"
             f"🔁 {last_regime} ➜ {curr_regime}\n"
             f"📌 <b>{signal_to_write}</b>\n"
             f"💰 Close: {last_close:.2f}\n"
             f"⚙️ Mode: immédiat (sans hystérésis / sans confirmation)")
        print(msg); await send_telegram(msg)
        last_regime=curr_regime; last_signal_time_ms=last_time
    finally:
        conn.close()

async def init_db():
    try:
        if os.path.exists(db_path):
            os.remove(db_path); message="♻️ Base de données existante détectée et supprimée."
        else:
            message="🆕 Aucune base existante — création d'une nouvelle base de données..."
        print(message); await send_telegram(message)
        conn=sqlite3.connect(db_path); cursor=conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlc (
                time INTEGER PRIMARY KEY,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                rsi REAL, tema20 REAL, tema50 REAL, slope20 REAL, speed20 REAL,
                acceleration20 REAL, local_max20 REAL, local_min20 REAL, global_max20 REAL, global_min20 REAL
            )
        """)
        conn.commit(); conn.close()
        message="✅ Base de données initialisée avec succès\n📊 Table : OHLC + indicateurs techniques"
        print(message); await send_telegram(message)
    except sqlite3.Error as e:
        message=f"❌ ERREUR BASE DE DONNÉES\n🧱 SQLite : {e}"
        print(message); await send_telegram(message)

async def load_history():
    try:
        conn=sqlite3.connect(db_path); cursor=conn.cursor()
        message=f"📥 Récupération de {MAX_CANDLES} bougies historiques..."
        print(message); await send_telegram(message)
        klines=client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=MAX_CANDLES)
        message=f"📊 {len(klines)} bougies reçues de Binance"
        print(message); await send_telegram(message)
        for k in klines:
            cursor.execute("""INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)""",
                           (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        conn.commit()
        message="🧮 Calcul des indicateurs techniques en cours..."
        print(message); await send_telegram(message)
        df=pd.read_sql("SELECT * FROM ohlc ORDER BY time", conn)
        df=calculate_all_indicators(df)
        for _, row in df.iterrows():
            cursor.execute("""
                UPDATE ohlc
                SET rsi = ?, tema20 = ?, tema50 = ?, slope20 = ?, speed20 = ?,
                    acceleration20 = ?, local_max20 = ?, local_min20 = ?, global_max20 = ?, global_min20 = ?
                WHERE time = ?
            """, (
                None if pd.isna(row["rsi"]) else float(row["rsi"]),
                None if pd.isna(row["tema20"]) else float(row["tema20"]),
                None if pd.isna(row["tema50"]) else float(row["tema50"]),
                None if pd.isna(row["slope20"]) else float(row["slope20"]),
                None if pd.isna(row["speed20"]) else float(row["speed20"]),
                None if pd.isna(row["acceleration20"]) else float(row["acceleration20"]),
                None if pd.isna(row["local_max20"]) else float(row["local_max20"]),
                None if pd.isna(row["local_min20"]) else float(row["local_min20"]),
                None if pd.isna(row["global_max20"]) else float(row["global_max20"]),
                None if pd.isna(row["global_min20"]) else float(row["global_min20"]),
                int(row["time"])
            ))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM ohlc"); total=cursor.fetchone()[0]
        last_row=df.iloc[-1]; timestamp=datetime.fromtimestamp(last_row["time"]/1000)
        rsi_val=f"{last_row['rsi']:.2f}" if pd.notna(last_row["rsi"]) else "N/A"
        tema20_val=f"{last_row['tema20']:.2f}" if pd.notna(last_row["tema20"]) else "N/A"
        tema50_val=f"{last_row['tema50']:.2f}" if pd.notna(last_row["tema50"]) else "N/A"
        message=(f"✅ <b>{total} bougies chargées avec succès</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"🕐 Dernière bougie : {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                 f"💰 Close : {float(last_row['close']):.2f} USDC\n"
                 f"📊 RSI : {rsi_val}\n📈 TEMA20 : {tema20_val}\n📉 TEMA50 : {tema50_val}")
        print(message); await send_telegram(message)
        conn.close()
        await detect_and_emit_signal()
    except Exception as e:
        message=f"❌ Erreur lors du chargement : {e}"
        print(message); await send_telegram(message)

async def add_candle():
    try:
        conn=sqlite3.connect(db_path); cursor=conn.cursor()
        klines=client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1)
        for k in klines:
            cursor.execute("""INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)""",
                           (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        conn.commit()
        cursor.execute("SELECT * FROM ohlc ORDER BY time DESC LIMIT 200")
        rows=cursor.fetchall(); columns=[d[0] for d in cursor.description]
        df=pd.DataFrame(rows[::-1], columns=columns)
        df=calculate_all_indicators(df)
        last_row=df.iloc[-1]
        cursor.execute("""
            UPDATE ohlc
            SET rsi = ?, tema20 = ?, tema50 = ?, slope20 = ?, speed20 = ?,
                acceleration20 = ?, local_max20 = ?, local_min20 = ?, global_max20 = ?, global_min20 = ?
            WHERE time = ?
        """, (
            None if pd.isna(last_row["rsi"]) else float(last_row["rsi"]),
            None if pd.isna(last_row["tema20"]) else float(last_row["tema20"]),
            None if pd.isna(last_row["tema50"]) else float(last_row["tema50"]),
            None if pd.isna(last_row["slope20"]) else float(last_row["slope20"]),
            None if pd.isna(last_row["speed20"]) else float(last_row["speed20"]),
            None if pd.isna(last_row["acceleration20"]) else float(last_row["acceleration20"]),
            None if pd.isna(last_row["local_max20"]) else float(last_row["local_max20"]),
            None if pd.isna(last_row["local_min20"]) else float(last_row["local_min20"]),
            None if pd.isna(last_row["global_max20"]) else float(last_row["global_max20"]),
            None if pd.isna(last_row["global_min20"]) else float(last_row["global_min20"]),
            int(last_row["time"])
        ))
        conn.commit()
        timestamp=datetime.fromtimestamp(last_row["time"]/1000)
        message=(f"🆕 <b>Nouvelle bougie ajoutée</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                 f"🕐 Timestamp : {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                 f"💰 Close : {float(last_row['close']):.2f} USDC")
        print(message); await send_telegram(message)
        conn.close()
        await detect_and_emit_signal()
    except Exception as e:
        message=f"❌ Erreur lors de l'ajout : {e}"
        print(message); await send_telegram(message)

def interval_to_seconds(interval:str)->int:
    interval=interval.strip().lower()
    if interval.endswith("m"): return int(interval[:-1])*60
    if interval.endswith("h"): return int(interval[:-1])*3600
    if interval.endswith("d"): return int(interval[:-1])*86400
    return 60

async def wait_next_candle():
    sec=interval_to_seconds(INTERVAL)
    now=time.time()
    next_ts=((int(now)//sec)+1)*sec
    wait_seconds=max(0, next_ts-now)+2
    next_dt=datetime.fromtimestamp(next_ts)
    message=(f"⏰ Heure actuelle: {datetime.now().strftime('%H:%M:%S')}\n"
             f"⏳ Prochaine bougie à: {next_dt.strftime('%H:%M:%S')}\n"
             f"⏱️ Attente de {int(wait_seconds)} secondes...")
    print(message); await send_telegram(message)
    await asyncio.sleep(wait_seconds)

async def show_infos():
    try:
        conn=sqlite3.connect(db_path); cursor=conn.cursor()
        cursor.execute("SELECT time, close, tema20, tema50 FROM ohlc ORDER BY time DESC LIMIT 1")
        row=cursor.fetchone()
        if row:
            timestamp=datetime.fromtimestamp(row[0]/1000); close=float(row[1]); tema20, tema50=row[2], row[3]
            regime=_regime_from_tema(tema20, tema50)
            signal="LONG 📈" if regime=="LONG" else ("SHORT 📉" if regime=="SHORT" else "WAIT ⏳")
            message=(f"⚡ Régime : {signal}\n🕐 {timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                     f"💰 Close : {close:.2f}\n"
                     f"⚙️ Détection: immédiate (sans hystérésis / sans confirmation)\n"
                     f"📄 signal.txt : {SIGNAL_FILE}")
        else:
            message="⚠️ Pas de bougie trouvée dans la base de données."
        print(message); await send_telegram(message)
        conn.close()
    except Exception as e:
        print(f"❌ Erreur show_infos : {e}")
        await send_telegram(f"❌ Erreur show_infos : {e}")

async def bot_async():
    write_signal("")
    await init_db()
    message="🤖 BOT SIGNALS LANCÉ\n━━━━━━━━━━━━━━━━━━━━━━\n⚙️ Initialisation..."
    print(message); await send_telegram(message)
    await load_history()
    message=("🚀 <b>Bot opérationnel</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
             f"📈 Symbole    : {SYMBOL}\n"
             f"⏱️ Timeframe  : {INTERVAL}\n"
             "📊 Stratégie : TEMA 20 / TEMA 50\n"
             "⚙️ Mode: immédiat (sans hystérésis / sans confirmation)\n"
             f"📝 Signal file : {SIGNAL_FILE}")
    print(message); await send_telegram(message)
    while True:
        await show_infos()
        await wait_next_candle()
        await add_candle()

def bot(): asyncio.run(bot_async())

if __name__=="__main__":
    try: bot()
    except KeyboardInterrupt:
        stop_message=("🛑 <b>Arrêt manuel du bot</b>\n━━━━━━━━━━━━━━━━━━━━━━\n📴 Exécution interrompue par l'utilisateur")
        print("\n🛑 Bot arrêté proprement")
        asyncio.run(send_telegram(stop_message))
