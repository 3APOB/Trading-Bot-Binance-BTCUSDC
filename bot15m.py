from binance.client import Client
import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from telegram import Bot
import asyncio
from io import BytesIO
# =========================
# 🔐 CONFIG
# =========================
API_KEY = ""
API_SECRET = ""
# 🔔 Configuration Telegram
TELEGRAM_TOKEN = ""  # Obtenir via @BotFather
TELEGRAM_CHAT_ID = ""  # Votre ID utilisateur
SYMBOL = "BTCUSDC"
INTERVAL = "15m"
DB_NAME = "db15M.db"
MAX_CANDLES = 1000
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
client = Client(API_KEY, API_SECRET)
# =========================
# 📦 PORTFOLIO SIMULÉ
# =========================
wallet_usdc = 100.0
wallet_btc = 0.0
fee_trade_buy = 1.001
fee_trade_sell = 0.999
qtt_trade_btc = 0.0001

achat_prices = []   # prix auxquels tu as acheté
vente_prices = []   # prix auxquels tu as vendu
seuil = 0.03
position = "NONE"
positions_achat = []  # mémoire des trades
positions_vente = []  # mémoire des trades
positions = []

async def send_telegram(message):
    bot = Bot(TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
    print("Message envoyé !")

async def init_db():
    """Initialise la base de données"""
    # création du tableau de bdd de ohlc et inidcator du btcusdc
    try:
        # Vérifie si le fichier existe, et le supprime s'il existe
        if os.path.exists(db_path):
            os.remove(db_path)
            message = "⚠️ Fichier existant supprimé."
            print(message)
            await send_telegram(message)
        else:
            message = f"⚠️ Fichier en création..."
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
                slope REAL,
                speed REAL,
                acceleration REAL,
                local_max REAL,
                local_min REAL,
                global_max REAL,
                global_min REAL,
                bb_high REAL,
                bb_low REAL,
                tenkan REAL,
                kijun REAL,
                senkou_a REAL,
                senkou_b REAL
            )
        """)
        conn.commit()
        conn.close()
        message = f"⚠️ ✅ Base de données créée🆕"
        print(message)
        await send_telegram(message)
    except sqlite3.Error as e:
        message = f"❌ Erreur SQLite : {e}"
        print(message)
        await send_telegram(message)

async def load_history():
    """Récupère et stocke les nouvelles bougies"""
    try:
        # connection à bdd
        conn = sqlite3.connect(db_path)
        # requête sql
        cursor = conn.cursor()
        message = f"📥 Récupération des bougies..."
        print(message)
        await send_telegram(message)
        # récuparartion des 1000 dernières candles
        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=MAX_CANDLES)
        for kline in klines: 
            cursor.execute("""
                INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                float(kline[0]),
                float(kline[1]),
                float(kline[2]),
                float(kline[3]),
                float(kline[4]),
                float(kline[5])
            ))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM ohlc")
        total = cursor.fetchone()[0]
        message = f"✅  bougies (Total: {total})ℹ️"
        print(message)
        await send_telegram(message)
        df = pd.read_sql("SELECT time, close FROM ohlc ORDER BY time", conn)
        nb_candles = len(df)
        # --- TEMA 20 ---
        if nb_candles >= 60:
            ema1_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema2_20 = ema1_20.ewm(span=20, adjust=False).mean()
            ema3_20 = ema2_20.ewm(span=20, adjust=False).mean()
            df['tema20'] = 3 * ema1_20 - 3 * ema2_20 + ema3_20
        else:
            df['tema20'] = None
            print("⚠️ Pas assez de données pour TEMA 20")
        # --- TEMA 50 ---
        if nb_candles >= 150:
            ema1_50 = df['close'].ewm(span=50, adjust=False).mean()
            ema2_50 = ema1_50.ewm(span=50, adjust=False).mean()
            ema3_50 = ema2_50.ewm(span=50, adjust=False).mean()
            df['tema50'] = 3 * ema1_50 - 3 * ema2_50 + ema3_50
        else:
            df['tema50'] = None
            print("⚠️ Pas assez de données pour TEMA 50")
        df.loc[df.index < 60, 'tema20'] = None
        df.loc[df.index < 150, 'tema50'] = None
        # =====================
        # 💾 UPDATE SQL
        # =====================
        for _, row in df.iterrows():
            cursor.execute("""
                UPDATE ohlc
                SET tema20 = ?, tema50 = ?
                WHERE time = ?
            """, (
                None if pd.isna(row['tema20']) else float(row['tema20']),
                None if pd.isna(row['tema50']) else float(row['tema50']),
                int(row['time'])
            ))
        conn.commit()
        message = "📈 Calcul TEMA terminé"
        print(message)
        await send_telegram(message)
        conn.close()
    except Exception as e:
        print(f"❌ Erreur OHLC : {e}")

async def add_candle():
    """Récupère et stocke la dernière candle"""
    try:
        # connection à bdd
        conn = sqlite3.connect(db_path)
        # requête sql
        cursor = conn.cursor()
        message = f"📥 Récupération de la derniere candle"
        print(message)
        await send_telegram(message)
        # récuparartion la dernières candles
        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1)
        for kline in klines:
            cursor.execute("""
                INSERT OR IGNORE INTO ohlc (time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                float(kline[0]),
                float(kline[1]),
                float(kline[2]),
                float(kline[3]),
                float(kline[4]),
                float(kline[5])
            ))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM ohlc")
        total = cursor.fetchone()[0]
        message = f"✅  nouvelles bougies (Total: {total})ℹ️"
        print(message)
        await send_telegram(message)
        cursor.execute("SELECT time, close FROM ohlc ORDER BY time DESC LIMIT 200")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows[::-1], columns=['time', 'close'])
        nb_candles = len(df)
        # --- TEMA 20 ---
        if nb_candles >= 60:
            ema1_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema2_20 = ema1_20.ewm(span=20, adjust=False).mean()
            ema3_20 = ema2_20.ewm(span=20, adjust=False).mean()
            df['tema20'] = 3 * ema1_20 - 3 * ema2_20 + ema3_20
        else:
            df['tema20'] = None
            print("⚠️ Pas assez de données pour TEMA 20")
        # --- TEMA 50 ---
        if nb_candles >= 150:
            ema1_50 = df['close'].ewm(span=50, adjust=False).mean()
            ema2_50 = ema1_50.ewm(span=50, adjust=False).mean()
            ema3_50 = ema2_50.ewm(span=50, adjust=False).mean()
            df['tema50'] = 3 * ema1_50 - 3 * ema2_50 + ema3_50
        else:
            df['tema50'] = None
            print("⚠️ Pas assez de données pour TEMA 50")
        # =====================
        # 💾 UPDATE SQL
        # =====================
        last_row = df.iloc[-1]
        cursor.execute("""
            UPDATE ohlc
            SET tema20 = ?, tema50 = ?
            WHERE time = ?
        """, (  
            None if pd.isna(last_row['tema20']) else float(last_row['tema20']),
            None if pd.isna(last_row['tema50']) else float(last_row['tema50']),
            int(last_row['time'])
        ))
        conn.commit()
        message = "📈 Calcul TEMA terminé"
        print(message)
        await send_telegram(message)
        conn.close()
    except Exception as e:
        print(f"❌ Erreur OHLC : {e}")

async def wait_next_candle():
    """Attend la prochaine bougie de 15 minutes"""
    now = datetime.now()
    # Calcul des minutes jusqu'au prochain multiple de 15
    current_minute = now.minute
    next_candle_minute = ((current_minute // 15) + 1) * 15
    # Si on dépasse 60, on passe à l'heure suivante
    if next_candle_minute >= 60:
        next_candle_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_candle_time = now.replace(minute=next_candle_minute, second=0, microsecond=0)
    wait_seconds = (next_candle_time - now).total_seconds()
    message = f"⏰ Heure actuelle: {now.strftime('%H:%M:%S')}"
    print(message)
    await send_telegram(message)
    message = f"⏳ Prochaine bougie à: {next_candle_time.strftime('%H:%M:%S')}"
    print(message)
    await send_telegram(message)
    message = f"⏱️  Attente de {int(wait_seconds)} secondes..."
    print(message)
    await send_telegram(message)
    time.sleep(wait_seconds + 5)  # +5 secondes de sécurité pour être sûr que la bougie est complète

async def show_infos():
    global wallet_btc, wallet_usdc
    """Affiche le prix de la dernière bougie et les valeurs TEMA 20 et TEMA 50, et indique LONG/SHORT"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Récupère la dernière bougie
        cursor.execute("SELECT time, close, tema20, tema50 FROM ohlc ORDER BY time DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            timestamp = datetime.fromtimestamp(row[0] / 1000)  # Convert ms -> datetime
            close = row[1]
            tema20 = row[2]
            tema50 = row[3]
            # Déterminer le signal LONG/SHORT
            if tema20 is not None and tema50 is not None:
                if tema20 > tema50:
                    signal = "LONG 📈"
                elif tema20 < tema50:
                    signal = "SHORT 📉"
                else:
                    signal = "NEUTRE ⚪"
            else:
                signal = "Indéterminé ❌"
            total_wallet_brut = wallet_usdc + (wallet_btc*close)
            total_wallet_net = wallet_usdc + (wallet_btc*close*0.99)
            message = (
                f"📊 Dernière bougie : {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"💵 Prix : {close:.2f} USDC\n"
                f"📈 TEMA 20 : {tema20:.2f} {'❌' if tema20 is None else ''}\n"
                f"📊 TEMA 50 : {tema50:.2f} {'❌' if tema50 is None else ''}\n"
                f"⚡ Signal : {signal}\n"
                f"wallet : {wallet_usdc} $, {wallet_btc} BTC\n"
                f"Total Wallet: brut : {total_wallet_brut} usdc, net {total_wallet_net}"
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

# Calcul du nombre de positions possibles
async def calculate_available_positions():
    """Calcule combien de nouvelles positions peuvent être ouvertes"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT close FROM ohlc ORDER BY time DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return 0, 0
        
        price = row[0]
        
        # USDC libre = wallet total - USDC engagé dans positions actives
        usdc_engaged = sum(
            pos["qty"] * pos["buy_price"] * fee_trade_buy 
            for pos in positions 
            if pos["state"] == "WAIT_SELL" and pos["buy_price"] is not None
        )
        usdc_free = wallet_usdc - usdc_engaged
        
        # Nouvelles positions BUY possibles
        cost_per_buy = qtt_trade_btc * price * fee_trade_buy
        new_buy_positions = int(usdc_free / cost_per_buy) if cost_per_buy > 0 and usdc_free > 0 else 0
        
        # Nouvelles positions SELL possibles (BTC libre)
        btc_free = wallet_btc
        new_sell_positions = int(btc_free / qtt_trade_btc) if qtt_trade_btc > 0 else 0
        
        return new_buy_positions, new_sell_positions
        
    except Exception as e:
        print(f"❌ Erreur calculate_available_positions: {e}")
        return 0, 0


async def init_positions():
    """Initialise les positions en fonction du wallet disponible"""
    global positions
    
    positions = []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT close FROM ohlc ORDER BY time DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            price = 100000  # Prix estimé par défaut
        else:
            price = row[0]
        
        # Positions possibles avec USDC
        cost_per_position = qtt_trade_btc * price * fee_trade_buy
        max_buy_positions = int(wallet_usdc / cost_per_position)
        
        for i in range(max_buy_positions):
            positions.append({
                "id": f"B{i}",
                "state": "WAIT_BUY",
                "qty": qtt_trade_btc,
                "buy_price": None,
                "sell_price": None,
                "cycles": 0
            })
        
        # Positions possibles avec BTC stock
        max_sell_positions = int(wallet_btc / qtt_trade_btc) if qtt_trade_btc > 0 else 0
        
        for i in range(max_sell_positions):
            positions.append({
                "id": f"S{i}",
                "state": "WAIT_SELL",
                "qty": qtt_trade_btc,
                "buy_price": None,  # Stock initial sans prix d'achat
                "sell_price": None,
                "cycles": 0
            })
        
        message = (f"📊 Init: {max_buy_positions} positions WAIT_BUY (USDC) | "
                  f"{max_sell_positions} positions WAIT_SELL (BTC stock)")
        print(message)
        await send_telegram(message)
        
    except Exception as e:
        message = f"❌ Erreur init_positions: {e}"
        print(message)
        await send_telegram(message)
    
    return positions


async def check_tema():
    """Analyse TEMA et retourne le signal de trading"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT close, tema20, tema50
            FROM ohlc
            WHERE tema20 IS NOT NULL AND tema50 IS NOT NULL
            ORDER BY time DESC
            LIMIT 5
        """)
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 5:
            print("⏳ Pas assez de données TEMA")
            return "UNKNOWN"
        
        # Extraction des données
        close = [row[0] for row in rows]
        tema20 = [row[1] for row in rows]
        tema50 = [row[2] for row in rows]
        
        # Métriques
        ecart_absolue_20_50 = tema20[0] - tema50[0]
        ecart_relatif_20_50 = (ecart_absolue_20_50 / tema50[0]) * 100
        
        prix_vs_tema20 = close[0] - tema20[0]
        prix_vs_tema20_pct = (prix_vs_tema20 / tema20[0]) * 100
        
        slope_short_20 = tema20[0] - tema20[1]
        slope_short_50 = tema50[0] - tema50[1]
        
        acceleration_20 = slope_short_20 - (tema20[1] - tema20[2])
        
        prix_momentum_court = close[0] - close[1]
        
        # Détection patterns
        is_valley = tema20[2] > tema20[1] < tema20[0]  # Creux
        is_peak = tema20[2] < tema20[1] > tema20[0]    # Sommet
        
        # Détection croisements
        croisement_haussier = (tema20[1] <= tema50[1] and tema20[0] > tema50[0])
        croisement_baissier = (tema20[1] >= tema50[1] and tema20[0] < tema50[0])
        
        debut_long = (croisement_haussier and slope_short_20 > 0 and 
                      acceleration_20 > 0 and close[0] > tema20[0])
        
        fin_long_debut_short = (croisement_baissier and slope_short_20 < 0 and 
                                acceleration_20 < 0 and close[0] < tema20[0])
        
        # État actuel
        if ecart_absolue_20_50 > 0:
            etat = "LONG"
        elif ecart_absolue_20_50 < 0:
            etat = "SHORT"
        else:
            etat = "FLAT"
        
        # Affichage
        message = (
            f"📊 ÉTAT: {etat} | Prix: {close[0]:.2f}\n"
            f"🎯 Prix vs TEMA20: {prix_vs_tema20_pct:+.3f}%\n"
            f"📏 Écart TEMA20/50: {ecart_relatif_20_50:.3f}%\n"
            f"📐 Pente TEMA20: {slope_short_20:.2f}\n"
            f"⚡ Accélération: {acceleration_20:.2f}\n"
            f"🚀 Momentum: {prix_momentum_court:+.2f}"
        )
        print(message)
        await send_telegram(message)
        
        # === SIGNAUX DE TRADING ===
        
        # SIGNAL BUY: Début de LONG (fin de gros SHORT)
        if debut_long:
            message = f"🟢 DÉBUT DE LONG détecté à {close[0]:.2f}"
            print(message)
            await send_telegram(message)
            return "BUY"
        
        # SIGNAL BUY: Rebond en creux pendant LONG
        if etat == "LONG" and is_valley and slope_short_20 > 0:
            message = f"🟢 REBOND EN CREUX détecté à {close[0]:.2f}"
            print(message)
            await send_telegram(message)
            return "BUY"
        
        # SIGNAL SELL: Fin de LONG / Début de SHORT
        if fin_long_debut_short:
            message = f"🔴 FIN DE LONG / DÉBUT SHORT détecté à {close[0]:.2f}"
            print(message)
            await send_telegram(message)
            return "SELL"
        
        # SIGNAL SELL: Sommet pendant SHORT
        if etat == "SHORT" and is_peak and slope_short_20 < 0:
            message = f"🔴 SOMMET EN SHORT détecté à {close[0]:.2f}"
            print(message)
            await send_telegram(message)
            return "SELL"
        
        print("⏸️ Aucun signal fort")
        return "WAIT"
    
    except Exception as e:
        message = f"❌ Erreur check TEMA: {e}"
        print(message)
        await send_telegram(message)
        return "UNKNOWN"


async def order(signal):
    """Gère les ordres - Positions en cycle continu A-V-A-V ou V-A-V-A"""
    global wallet_usdc, wallet_btc, positions
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT close FROM ohlc ORDER BY time DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            raise ValueError("Aucun prix disponible")
        
        price = row[0]
        
        # === GESTION DES ACHATS ===
        if signal == "BUY":
            # 1. OUVRIR nouvelle position si bénéfices disponibles
            new_buy_possible, _ = await calculate_available_positions()
            if new_buy_possible > 0:
                cost = qtt_trade_btc * price * fee_trade_buy
                
                if wallet_usdc >= cost:
                    wallet_usdc -= cost
                    wallet_btc += qtt_trade_btc
                    
                    new_id = f"B{len([p for p in positions if 'B' in str(p['id'])])}"
                    new_pos = {
                        "id": new_id,
                        "state": "WAIT_SELL",
                        "qty": qtt_trade_btc,
                        "buy_price": price,
                        "sell_price": None,
                        "cycles": 0
                    }
                    positions.append(new_pos)
                    
                    message = f"🆕 OUVERTURE position {new_id} @ {price:.2f} → cycle A-V"
                    print(message)
                    await send_telegram(message)
            
            # 2. RACHETER positions en WAIT_BUY (cycle continu)
            for pos in positions:
                if pos["state"] == "WAIT_BUY":
                    cost = pos["qty"] * price * fee_trade_buy
                    
                    if wallet_usdc >= cost:
                        wallet_usdc -= cost
                        wallet_btc += pos["qty"]
                        pos["state"] = "WAIT_SELL"
                        pos["buy_price"] = price
                        
                        message = f"🔁 RACHAT position {pos['id']} @ {price:.2f} → WAIT_SELL (cycle #{pos['cycles']})"
                        print(message)
                        await send_telegram(message)
                        break
        
        # === GESTION DES VENTES ===
        elif signal == "SELL":
            # Vendre positions WAIT_SELL qui ont atteint le seuil
            for pos in positions:
                if pos["state"] == "WAIT_SELL":
                    # Si stock initial (pas de buy_price), vendre directement
                    # Sinon vérifier le seuil
                    should_sell = (pos["buy_price"] is None or 
                                  price >= pos["buy_price"] * (1 + seuil))
                    
                    if should_sell and wallet_btc >= pos["qty"]:
                        revenue = pos["qty"] * price * fee_trade_sell
                        wallet_btc -= pos["qty"]
                        wallet_usdc += revenue
                        
                        if pos["buy_price"]:
                            gain = (price - pos["buy_price"]) * pos["qty"] * fee_trade_sell
                            message = (f"🔴 VENTE position {pos['id']} @ {price:.2f} "
                                      f"(acheté @ {pos['buy_price']:.2f}, gain: {gain:.2f} USDC)")
                        else:
                            message = f"🔴 VENTE stock position {pos['id']} @ {price:.2f}"
                        
                        pos["state"] = "WAIT_BUY"
                        pos["sell_price"] = price
                        pos["cycles"] += 1
                        
                        message += f" → WAIT_BUY (cycle #{pos['cycles']})"
                        print(message)
                        await send_telegram(message)
                        break
        
        # Vérifier nouvelles positions possibles
        new_buy, new_sell = await calculate_available_positions()
        
        # Affichage wallet
        message = (f"💰 Wallet: {wallet_btc:.8f} BTC, {wallet_usdc:.2f} USDC\n"
                  f"🆕 Nouvelles positions possibles: {new_buy} achats")
        print(message)
        await send_telegram(message)
        
        # Statistiques positions
        wait_buy = sum(1 for p in positions if p["state"] == "WAIT_BUY")
        wait_sell = sum(1 for p in positions if p["state"] == "WAIT_SELL")
        total_cycles = sum(p["cycles"] for p in positions)
        total_positions = len(positions)
        
        stats = (f"📊 {total_positions} positions: {wait_buy} WAIT_BUY | "
                f"{wait_sell} WAIT_SELL | {total_cycles} cycles totaux")
        print(stats)
        
    except Exception as e:
        message = f"❌ Erreur order: {e}"
        print(message)
        await send_telegram(message)

async def bot_async():
    await init_db()
    await load_history()
    message = "\n🤖 BOT DÉMARRÉ"
    print(message)
    await send_telegram(message)
    message = f"🚀 <b>Bot démarré</b>\n💰 wallet: {wallet_usdc} USDC, {wallet_btc} BTC\n📈 Symbole: {SYMBOL}"
    print (message)
    await send_telegram(message)
    await show_infos()
    await init_positions()
    while True:
        await wait_next_candle()
        await add_candle()
        await show_infos()
        signal = await check_tema()

        if signal in ["BUY", "SELL"]:
            await order(signal)

def bot():
    asyncio.run(bot_async())

# =========================
if __name__ == "__main__":
    try:
        bot()
    except KeyboardInterrupt:
        print("\n🛑 Bot arrêté proprement")
        asyncio.run(send_telegram("🛑 Bot arrêté"))
