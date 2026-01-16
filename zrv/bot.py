"""
Zarov Trading Bot - Binance
Author: Zarov.
Version: 1.0.4 - BUGS RÉSOLUS ✅
"""
import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Bot
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging
from contextlib import contextmanager
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "config.env"))

# ==================== CONFIG ====================
@dataclass
class TradingConfig:
    """Centralized configuration with validation"""
    # API Credentials
    API_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    API_SECRET: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    
    # Trading Parameters
    SYMBOL: str = "BTCUSDC"
    INTERVAL: str = "1m"
    INITIAL_CAPITAL_USDC: float = 100.0
    INITIAL_CAPITAL_BTC: float = 0.0
    POSITION_SIZE: float = 0.0001
    FEE_RATE: float = 0.01
    
    # Strategy Parameters
    SEUIL: float = 2.2
    MAX_CANDLES: int = 1000
    
    # Technical Indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 32
    RSI_OVERBOUGHT: int = 68
    TEMA_FAST: int = 20
    TEMA_SLOW: int = 50
    
    # Database
    DB_NAME: str = "db.db"
    
    def __post_init__(self):
        """Validate configuration"""
        missing = []
        if not self.API_KEY:
            missing.append("BINANCE_API_KEY")
        if not self.API_SECRET:
            missing.append("BINANCE_SECRET")
        if not self.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not self.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        
        if missing:
            print("\n" + "="*60)
            print("ERROR: Missing credentials in config.env file")
            print("="*60)
            print(f"Missing variables: {', '.join(missing)}")
            print("\nCreate a file named 'config.env' with:")
            print("-"*60)
            print("BINANCE_API_KEY=your_api_key_here")
            print("BINANCE_SECRET=your_secret_here")
            print("TELEGRAM_TOKEN=your_telegram_token_here")
            print("TELEGRAM_CHAT_ID=your_chat_id_here")
            print("-"*60)
            raise ValueError(f"Missing credentials: {', '.join(missing)}")
        
        if self.POSITION_SIZE <= 0:
            raise ValueError("Invalid trading parameters")

# ==================== ENUMS ====================
class PositionState(Enum):
    """État de la position dans le cycle"""
    NO_POSITION = 0
    IN_POSITION = 1

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

# ==================== LOGGING ====================
def setup_logging():
    """Configure professional logging (Windows compatible)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== WALLET MANAGEMENT ====================
@dataclass
class Wallet:
    """Wallet management - Cycle A-V-A-V avec seuil"""
    cash: float
    btc: float
    state: PositionState = PositionState.NO_POSITION
    entry_price: Optional[float] = None
    trades: List[Dict] = field(default_factory=list)
    peak_value: float = field(init=False)
    
    def __post_init__(self):
        if self.btc > 0:
            self.state = PositionState.IN_POSITION
        else:
            self.state = PositionState.NO_POSITION
        self.peak_value = self.cash
    
    def total_value(self, current_price: float) -> float:
        return self.cash + (self.btc * current_price)
    
    def unrealized_pnl_percent(self, current_price: float) -> float:
        """Calcul du PnL en % depuis l'entrée"""
        if not self.entry_price or self.state == PositionState.NO_POSITION:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100
    
    def buy(self, price: float, size: float, fee: float):
        """Achat de BTC (entrée en position)"""
        if self.state == PositionState.IN_POSITION:
            logger.warning("Already in position, cannot buy")
            return False
        
        cost = price * size * (1 + fee)
        if cost > self.cash:
            logger.warning(f"Insufficient cash: {self.cash:.2f} < {cost:.2f}")
            return False
        
        self.cash -= cost
        self.btc += size
        self.entry_price = price
        self.state = PositionState.IN_POSITION
        
        logger.info(f"💰 BUY {size:.6f} BTC @ {price:.2f} | Cost: {cost:.2f} USDC")
        return True
    
    def sell(self, price: float, fee: float):
        """Vente de tout le BTC (sortie de position)"""
        if self.state == PositionState.NO_POSITION:
            logger.warning("No position to sell")
            return False
        
        if self.btc <= 0:
            logger.warning("No BTC to sell")
            return False
        
        pnl_percent = self.unrealized_pnl_percent(price)
        proceeds = self.btc * price * (1 - fee)
        sold_amount = self.btc
        
        self.cash += proceeds
        self.btc = 0.0
        
        self.trades.append({
            'timestamp': datetime.now(),
            'action': 'SELL',
            'entry_price': self.entry_price,
            'exit_price': price,
            'amount': sold_amount,
            'pnl_percent': pnl_percent,
            'proceeds': proceeds
        })
        
        logger.info(f"💵 SELL {sold_amount:.6f} BTC @ {price:.2f} | PnL: {pnl_percent:+.2f}% | Proceeds: {proceeds:.2f} USDC")
        
        self.entry_price = None
        self.state = PositionState.NO_POSITION
        
        return True
    
    def metrics(self, current_price: float) -> Dict:
        total = self.total_value(current_price)
        self.peak_value = max(self.peak_value, total)
        
        winning_trades = [t for t in self.trades if t['pnl_percent'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_percent'] < 0]
        
        return {
            'total_value': total,
            'cash': self.cash,
            'btc_holdings': self.btc,
            'state': self.state.name,
            'unrealized_pnl': self.unrealized_pnl_percent(current_price),
            'total_trades': len(self.trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'avg_win': np.mean([t['pnl_percent'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl_percent'] for t in losing_trades]) if losing_trades else 0,
            'max_drawdown': ((self.peak_value - total) / self.peak_value * 100) if self.peak_value > 0 else 0
        }

# ==================== TECHNICAL INDICATORS ====================
class TechnicalAnalysis:
    """Optimized technical indicators"""
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def tema(close: pd.Series, period: int) -> pd.Series:
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema3
    
    @staticmethod
    def momentum_indicators(tema: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        slope = tema.diff(period) / period
        acceleration = slope.diff(period) / period
        return {
            'slope': slope,
            'acceleration': acceleration,
            'local_max': tema.rolling(period, center=True).max(),
            'local_min': tema.rolling(period, center=True).min()
        }
    
    @staticmethod
    def compute_all(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
        """Compute all indicators efficiently"""
        df = df.copy()
        df['rsi'] = TechnicalAnalysis.rsi(df['close'], config.RSI_PERIOD)
        df['tema_fast'] = TechnicalAnalysis.tema(df['close'], config.TEMA_FAST)
        df['tema_slow'] = TechnicalAnalysis.tema(df['close'], config.TEMA_SLOW)
        
        momentum = TechnicalAnalysis.momentum_indicators(df['tema_fast'], config.TEMA_FAST)
        for key, val in momentum.items():
            df[key] = val
        
        return df

# ==================== DATABASE ====================
class DataManager:
    """Efficient database operations with context manager"""
    
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS candles (
            timestamp INTEGER PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            rsi REAL, tema_fast REAL, tema_slow REAL,
            slope REAL, acceleration REAL,
            local_max REAL, local_min REAL
        )
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self.SCHEMA)
            conn.commit()
    
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_or_update_candle(self, row: pd.Series):
        """Insert or update single candle"""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candles 
                (timestamp, open, high, low, close, volume, rsi, tema_fast, tema_slow, 
                 slope, acceleration, local_max, local_min)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row['timestamp']), row['open'], row['high'], row['low'], 
                row['close'], row['volume'], row.get('rsi'), row.get('tema_fast'), 
                row.get('tema_slow'), row.get('slope'), row.get('acceleration'),
                row.get('local_max'), row.get('local_min')
            ))
            conn.commit()
    
    def bulk_insert(self, df: pd.DataFrame):
        """Insert multiple candles with conflict handling"""
        with self._connect() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO candles 
                        (timestamp, open, high, low, close, volume, rsi, tema_fast, tema_slow,
                         slope, acceleration, local_max, local_min)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        int(row['timestamp']), row['open'], row['high'], row['low'],
                        row['close'], row['volume'], row.get('rsi'), row.get('tema_fast'),
                        row.get('tema_slow'), row.get('slope'), row.get('acceleration'),
                        row.get('local_max'), row.get('local_min')
                    ))
                except Exception as e:
                    logger.debug(f"Skip duplicate timestamp {row['timestamp']}: {e}")
            conn.commit()
    
    def get_latest(self, limit: int = 200) -> pd.DataFrame:
        """Retrieve latest candles"""
        with self._connect() as conn:
            df = pd.read_sql(
                f"SELECT * FROM candles ORDER BY timestamp DESC LIMIT {limit}",
                conn
            )
            if len(df) == 0:
                return pd.DataFrame()
            return df.iloc[::-1].reset_index(drop=True)
    
    def get_last_timestamp(self) -> Optional[int]:
        """Get last stored timestamp"""
        with self._connect() as conn:
            result = conn.execute("SELECT MAX(timestamp) FROM candles").fetchone()
            return result[0] if result[0] else None

# ==================== TRADING STRATEGY ====================
class ProStrategy:
    """Strategy: Cycle A-V-A-V avec seuil de 2.2%"""
    
    def __init__(self, config: TradingConfig, wallet: Wallet):
        self.cfg = config
        self.wallet = wallet
    
    def analyze(self, current: pd.Series, previous: pd.Series) -> Optional[Dict]:
        """Generate trading signals basé sur le cycle A-V"""
        price = current['close']
        
        # Vérifier que les indicateurs sont valides
        if pd.isna(current['rsi']) or pd.isna(current['slope']):
            logger.debug("Indicators not ready, skipping signal generation")
            return None
        
        if self.wallet.state == PositionState.IN_POSITION:
            pnl = self.wallet.unrealized_pnl_percent(price)
            
            if pnl >= self.cfg.SEUIL:
                return {
                    'action': SignalType.SELL,
                    'reason': f'TAKE_PROFIT ({pnl:.2f}%)',
                    'price': price
                }
            
            if self._is_strong_bearish(current, previous):
                return {
                    'action': SignalType.SELL,
                    'reason': f'BEARISH_SIGNAL (PnL: {pnl:.2f}%)',
                    'price': price
                }
        else:
            if self._is_strong_bullish(current, previous):
                return {
                    'action': SignalType.BUY,
                    'reason': 'BULLISH_SIGNAL',
                    'price': price
                }
        
        return None
    
    def _is_strong_bullish(self, curr: pd.Series, prev: pd.Series) -> bool:
        """Signal haussier fort pour ACHETER"""
        if any(pd.isna([curr['slope'], curr['acceleration'], curr['rsi'], prev['slope']])):
            return False
        
        if (prev['slope'] <= 0 and curr['slope'] > 0 and 
            curr['acceleration'] > 0 and curr['rsi'] < 50):
            return True
        
        if (curr['rsi'] < self.cfg.RSI_OVERSOLD and 
            curr['acceleration'] > 0 and
            not pd.isna(curr['local_min']) and
            curr['close'] < curr['local_min'] * 1.02):
            return True
        
        return False
    
    def _is_strong_bearish(self, curr: pd.Series, prev: pd.Series) -> bool:
        """Signal baissier fort pour VENDRE"""
        if any(pd.isna([curr['slope'], curr['acceleration'], curr['rsi'], prev['slope']])):
            return False
        
        if (prev['slope'] >= 0 and curr['slope'] < 0 and 
            curr['acceleration'] < 0 and curr['rsi'] > 50):
            return True
        
        if (curr['rsi'] > self.cfg.RSI_OVERBOUGHT and 
            curr['acceleration'] < 0 and
            not pd.isna(curr['local_max']) and
            curr['close'] > curr['local_max'] * 0.98):
            return True
        
        return False

# ==================== TRADING BOT ====================
class TradingBot:
    """Main bot orchestrator"""
    
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.telegram = Bot(config.TELEGRAM_TOKEN)
        self.db = DataManager(config.DB_NAME)
        self.wallet = Wallet(cash=config.INITIAL_CAPITAL_USDC, btc=config.INITIAL_CAPITAL_BTC)
        self.strategy = ProStrategy(config, self.wallet)
        self.last_processed_timestamp = None
        self.loop_counter = 0  # ✅ Compteur pour debug
    
    async def notify(self, message: str, level: str = "INFO"):
        """Send notifications"""
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
        formatted = f"{emoji.get(level, 'ℹ️')} {message}"
        try:
            await self.telegram.send_message(
                chat_id=self.cfg.TELEGRAM_CHAT_ID,
                text=formatted,
                parse_mode='HTML'
            )
            logger.info(formatted)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def initialize(self):
        """Setup and load historical data"""
        await self.notify("🚀 TRADING BOT STARTED", "SUCCESS")
        
        try:
            klines = self.client.get_klines(
                symbol=self.cfg.SYMBOL,
                interval=self.cfg.INTERVAL,
                limit=self.cfg.MAX_CANDLES
            )
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df = TechnicalAnalysis.compute_all(df, self.cfg)
        
        self.db.bulk_insert(df)
        
        # ✅ Initialiser avec le dernier timestamp
        self.last_processed_timestamp = int(df.iloc[-1]['timestamp'])
        
        initial_price = df.iloc[-1]['close']
        
        if self.wallet.btc > 0:
            self.wallet.entry_price = initial_price
        
        await self.notify(
            f"📊 Loaded {len(df)} candles | {self.cfg.SYMBOL} @ {initial_price:.2f}\n"
            f"Initial State: <b>{self.wallet.state.name}</b>\n"
            f"Cash: {self.wallet.cash:.2f} USDC | BTC: {self.wallet.btc:.6f}\n"
            f"Last timestamp: {datetime.fromtimestamp(self.last_processed_timestamp/1000).strftime('%H:%M:%S')}",
            "SUCCESS"
        )
    
    async def execute_trade(self, signal: Dict):
        """Execute trading logic"""
        if signal['action'] == SignalType.BUY:
            success = self.wallet.buy(
                signal['price'],
                self.cfg.POSITION_SIZE,
                self.cfg.FEE_RATE
            )
            
            if success:
                await self.notify(
                    f"🟢 <b>BUY</b> @ {signal['price']:.2f}\n"
                    f"Reason: {signal['reason']}\n"
                    f"Amount: {self.cfg.POSITION_SIZE:.6f} BTC",
                    "SUCCESS"
                )
        
        elif signal['action'] == SignalType.SELL:
            success = self.wallet.sell(
                signal['price'],
                self.cfg.FEE_RATE
            )
            
            if success:
                last_trade = self.wallet.trades[-1]
                await self.notify(
                    f"🔴 <b>SELL</b> @ {signal['price']:.2f}\n"
                    f"Reason: {signal['reason']}\n"
                    f"PnL: <b>{last_trade['pnl_percent']:+.2f}%</b>\n"
                    f"Proceeds: {last_trade['proceeds']:.2f} USDC",
                    "SUCCESS"
                )
    
    async def run(self):
        """✅ CORRIGÉ: Main trading loop avec logs détaillés"""
        await self.initialize()
        
        logger.info("🔄 Entering main loop...")
        
        while True:
            try:
                self.loop_counter += 1
                
                # ✅ Attendre la prochaine bougie (sans le +5 secondes inutile)
                await self._wait_next_candle()
                
                logger.info(f"🔁 Loop #{self.loop_counter} - Fetching new candle...")
                
                # Fetch new candle
                try:
                    klines = self.client.get_klines(
                        symbol=self.cfg.SYMBOL,
                        interval=self.cfg.INTERVAL,
                        limit=1
                    )
                except BinanceAPIException as e:
                    logger.error(f"Binance API error: {e}")
                    await asyncio.sleep(60)
                    continue
                
                new_timestamp = int(klines[0][0])
                new_time = datetime.fromtimestamp(new_timestamp/1000).strftime('%H:%M:%S')
                
                logger.info(f"📥 Received candle: {new_time} (timestamp: {new_timestamp})")
                
                # ✅ Vérifier si c'est une nouvelle bougie
                if new_timestamp <= self.last_processed_timestamp:
                    logger.info(f"⏭️ Candle {new_time} already processed (last: {datetime.fromtimestamp(self.last_processed_timestamp/1000).strftime('%H:%M:%S')})")
                    continue
                
                logger.info(f"✅ New candle detected: {new_time}")
                
                # Créer nouvelle bougie
                new_row = pd.Series({
                    'timestamp': float(new_timestamp),
                    'open': float(klines[0][1]),
                    'high': float(klines[0][2]),
                    'low': float(klines[0][3]),
                    'close': float(klines[0][4]),
                    'volume': float(klines[0][5])
                })
                
                logger.info(f"💹 Price: {new_row['close']:.2f} | Volume: {new_row['volume']:.2f}")
                
                # Récupérer historique + nouvelle bougie
                latest_df = self.db.get_latest(200)
                latest_df = pd.concat([latest_df, pd.DataFrame([new_row])], ignore_index=True)
                latest_df = TechnicalAnalysis.compute_all(latest_df, self.cfg)
                
                # Sauvegarder
                self.db.insert_or_update_candle(latest_df.iloc[-1])
                
                # Update timestamp
                self.last_processed_timestamp = new_timestamp
                
                # Analyze
                if len(latest_df) < 2:
                    logger.warning("Not enough data for analysis")
                    continue
                
                current = latest_df.iloc[-1]
                previous = latest_df.iloc[-2]
                
                # ✅ Log des indicateurs
                logger.info(f"📊 RSI: {current['rsi']:.2f} | TEMA slope: {current['slope']:.4f} | Accel: {current['acceleration']:.4f}")
                
                signal = self.strategy.analyze(current, previous)
                
                if signal:
                    logger.info(f"🎯 SIGNAL DETECTED: {signal['action'].value} - {signal['reason']}")
                    await self.execute_trade(signal)
                else:
                    logger.info(f"➖ No signal | State: {self.wallet.state.name}")
                    if self.wallet.state == PositionState.IN_POSITION:
                        pnl = self.wallet.unrealized_pnl_percent(current['close'])
                        logger.info(f"   💼 In position - Unrealized PnL: {pnl:+.2f}%")
                
                # ✅ Status toutes les 5 minutes (5 bougies)
                if self.loop_counter % 1 == 0:
                    metrics = self.wallet.metrics(current['close'])
                    
                    # Formatage conditionnel selon l'état
                    if self.wallet.state == PositionState.IN_POSITION:
                        pnl = metrics['unrealized_pnl']
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        target = self.cfg.SEUIL - pnl if pnl < self.cfg.SEUIL else 0
                        
                        status_msg = (
                            f"📊 <b>Position Update</b> #{self.loop_counter}\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"💰 Price: ${current['close']:.2f}\n"
                            f"{pnl_emoji} PnL: <b>{pnl:+.2f}%</b>\n"
                            f"🎯 Target: {self.cfg.SEUIL:.1f}% (need {target:+.2f}%)\n"
                            f"📈 RSI: {current['rsi']:.1f}\n"
                            f"📉 Momentum: {current['slope']:.2f}\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"💵 Value: ${metrics['total_value']:.2f}\n"
                            f"📊 Trades: {metrics['total_trades']} | WR: {metrics['win_rate']:.0f}%"
                        )
                    else:
                        status_msg = (
                            f"⏳ <b>Waiting for Entry</b> #{self.loop_counter}\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"💰 Price: ${current['close']:.2f}\n"
                            f"📈 RSI: {current['rsi']:.1f}\n"
                            f"📉 Momentum: {current['slope']:.2f}\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"💵 Cash: ${metrics['cash']:.2f}\n"
                            f"📊 Trades: {metrics['total_trades']} | WR: {metrics['win_rate']:.0f}%"
                        )
                    
                    await self.notify(status_msg, "INFO")
            
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                await self.notify(f"⚠️ Error: {str(e)}", "ERROR")
                await asyncio.sleep(60)
    
    async def _wait_next_candle(self):
        """✅ CORRIGÉ: Wait until next candle opens"""
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        wait_seconds = (next_minute - now).total_seconds()
        
        if wait_seconds > 0:
            logger.info(f"⏳ Waiting {wait_seconds:.1f}s until next candle ({next_minute.strftime('%H:%M:%S')})")
            await asyncio.sleep(wait_seconds)
        
        # ✅ Attendre 2 secondes après l'ouverture de la bougie pour être sûr
        await asyncio.sleep(2)

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        config = TradingConfig()
        bot = TradingBot(config)
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
