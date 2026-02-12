import os

def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1","true","yes","y","on")

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

RUN_KEY = os.getenv("RUN_KEY", "CHANGE_ME")

# TradingView webhook (optional)
TRADINGVIEW_WEBHOOK_KEY = os.getenv("TRADINGVIEW_WEBHOOK_KEY", RUN_KEY)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # legacy (admin DM chat id)
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", "")  # numeric user id; commands allowed only in private chat
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")  # @channelusername or numeric chat id for broadcasts

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "")
ALPACA_DATA_BASE_URL = os.getenv("ALPACA_DATA_BASE_URL", "")

# Scanner settings
UNIVERSE_MAX = env_int("UNIVERSE_MAX", 1500)
MIN_PRICE = env_float("MIN_PRICE", 2.0)
MAX_PRICE = env_float("MAX_PRICE", 250.0)
MIN_AVG_DOLLAR_VOL = env_float("MIN_AVG_DOLLAR_VOL", 2_000_000.0)
LOOKBACK_DAYS = env_int("LOOKBACK_DAYS", 60)
TOP_N = env_int("TOP_N", 80)  # return enough candidates; bot will send 7-10

# Execution / risk
BROKER = os.getenv("BROKER", "alpaca")  # only alpaca supported here
AUTO_TRADE = env_bool("AUTO_TRADE", False)
EXECUTE_TRADES = env_bool("EXECUTE_TRADES", False)
ALLOW_LIVE_TRADING = env_bool("ALLOW_LIVE_TRADING", False)

RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 0.25)  # percent of equity risked per trade
MAX_DAILY_TRADES = env_int("MAX_DAILY_TRADES", 2)

TP_R_MULT = env_float("TP_R_MULT", 1.8)
SL_ATR_MULT = env_float("SL_ATR_MULT", 1.5)

# Controls
SYMBOL_BATCH = env_int("SYMBOL_BATCH", 150)
REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 20)


# Trading time filters
SKIP_OPEN_MINUTES = env_int('SKIP_OPEN_MINUTES', 15)
SKIP_CLOSE_MINUTES = env_int('SKIP_CLOSE_MINUTES', 15)

# Market regime filter (e.g., SPY)
USE_MARKET_FILTER = env_bool('USE_MARKET_FILTER', True)
MARKET_SYMBOL = os.getenv('MARKET_SYMBOL', 'SPY')
MARKET_SMA_FAST = env_int('MARKET_SMA_FAST', 50)
MARKET_SMA_SLOW = env_int('MARKET_SMA_SLOW', 200)

# De-dup protections
BLOCK_IF_POSITION_OPEN = env_bool('BLOCK_IF_POSITION_OPEN', True)
BLOCK_IF_ORDER_OPEN = env_bool('BLOCK_IF_ORDER_OPEN', True)

# Reporting
SEND_DAILY_SUMMARY = env_bool('SEND_DAILY_SUMMARY', False)

# Local timezone (used for scheduled notifications)
LOCAL_TZ = os.getenv('LOCAL_TZ', 'Asia/Riyadh')



# AI Signal Filter (deterministic, no training)
AI_FILTER_ENABLED = env_bool('AI_FILTER_ENABLED', True)
AI_FILTER_MIN_SCORE = env_int('AI_FILTER_MIN_SCORE', 70)
AI_FILTER_SEND_REJECTS = env_bool('AI_FILTER_SEND_REJECTS', False)


# Signal evaluation & lightweight learning (no external ML deps)
SIGNAL_EVAL_DAYS = env_int('SIGNAL_EVAL_DAYS', 5)  # evaluate signals after N calendar days
ML_ENABLED = env_bool('ML_ENABLED', True)          # update lightweight weights from outcomes
ML_LEARNING_RATE = float(os.getenv('ML_LEARNING_RATE', '0.15'))

# Notification routing defaults (can be changed via Telegram menu)
NOTIFY_ROUTE_DEFAULT = os.getenv('NOTIFY_ROUTE_DEFAULT', 'dm')  # dm|group|both
NOTIFY_SILENT_DEFAULT = env_bool('NOTIFY_SILENT_DEFAULT', True)
