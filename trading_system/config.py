"""Central configuration — all optimised parameters loaded from .env."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Upstox credentials ────────────────────────────────────────────────────
UPSTOX_API_KEY: str = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET: str = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI: str = os.getenv("UPSTOX_REDIRECT_URI", "https://127.0.0.1")
UPSTOX_ACCESS_TOKEN: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")
UPSTOX_TOTP_SECRET: str = os.getenv("UPSTOX_TOTP_SECRET", "")
UPSTOX_MOBILE: str = os.getenv("UPSTOX_MOBILE", "")
UPSTOX_PIN: str = os.getenv("UPSTOX_PIN", "")

# ── Telegram ───────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Mode ───────────────────────────────────────────────────────────────────
PAPER_MODE: bool = os.getenv("PAPER_MODE", "True").lower() == "true"
UPSTOX_BASE_URL = (
        "https://sandbox-api.upstox.com" if PAPER_MODE else "https://api.upstox.com"
)

# ── Capital & Risk  ── OPTIMISED for Rs4,000+/day net target ──────────────
CAPITAL: float = float(os.getenv("TRADING_CAPITAL", "600000"))
MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "3000"))   # was 1000
RISK_PER_TRADE_PCT: float = 0.02           # 2% per trade  (was 1.5%)
TARGET_GROSS_DAILY: float = float(os.getenv("TARGET_GROSS_DAILY", "4500"))
TARGET_NET_DAILY: float = 4000.0           # minimum acceptable net/day

# ── Position sizing & entry ────────────────────────────────────────────────
MAX_POSITIONS: int = 3                     # concurrent positions (was 2)
SIGNAL_THRESHOLD: float = 0.55            # base threshold   (was 0.68)
SIGNAL_THRESHOLD_BULL: float = 0.58       # Bull-regime threshold
SIGNAL_THRESHOLD_SIDE: float = 0.55       # Sideways-regime threshold

# ── ATR multipliers (1.2 SL / 2.8 TP → 2.33 R:R) ─────────────────────────
ATR_MULT_SL: float = 1.2                  # tighter stop (was 1.5)
ATR_MULT_TP: float = 2.8                  # wider target (was 3.0 absolute)

# ── Charges ───────────────────────────────────────────────────────────────
INDIA_VIX_MAX: float = 20.0
CHARGES_PER_TRADE: float = 100.0          # estimated Rs100 per round trip

# ── Expanded watchlist: 15 liquid NSE F&O stocks ──────────────────────────
INSTRUMENT_MAP: dict[str, str] = {
        "RELIANCE":  "NSE_EQ|INE002A01018",
        "HDFCBANK":  "NSE_EQ|INE040A01034",
        "INFY":      "NSE_EQ|INE009A01021",
        "ICICIBANK": "NSE_EQ|INE090A01021",
        "TCS":       "NSE_EQ|INE467B01029",
        "KOTAKBANK": "NSE_EQ|INE237A01028",
        "LT":        "NSE_EQ|INE018A01030",
        "AXISBANK":  "NSE_EQ|INE238A01034",
        "SBIN":      "NSE_EQ|INE062A01020",
        "BAJFINANCE":"NSE_EQ|INE296A01024",
        "WIPRO":     "NSE_EQ|INE075A01022",
        "MARUTI":    "NSE_EQ|INE585B01010",
        "TITAN":     "NSE_EQ|INE280A01028",
        "ADANIENT":  "NSE_EQ|INE423A01024",
        "SUNPHARMA": "NSE_EQ|INE044A01036",
}
WATCHLIST: list[str] = list(INSTRUMENT_MAP.keys())

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────
WEIGHTS: dict[str, float] = {
        "technical":  0.40,
        "sentiment":  0.15,
        "fii_flow":   0.15,
        "gex":        0.20,
        "global_cue": 0.10,
}

# ── Schedule (IST 24-hour) ────────────────────────────────────────────────
SCHEDULE: dict[str, str] = {
        "token_refresh":  "08:00",
        "global_scan":    "08:30",
        "regime_check":   "08:45",
        "morning_brief":  "08:59",
        "market_open":    "09:15",
        "market_close":   "15:20",
        "eod_summary":    "15:30",
        "weekly_retrain": "Sunday 23:00",
}

# ── Paths ─────────────────────────────────────────────────────────────────
import pathlib
BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / "data_cache"
MODEL_DIR = BASE_DIR / "models" / "saved"
DB_PATH   = BASE_DIR / "trades.db"
LOG_PATH  = BASE_DIR / "trading_system.log"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
