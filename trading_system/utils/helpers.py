"""Shared utility functions used across the entire system."""

import logging
import pytz
import functools
import time
from datetime import datetime, date
from typing import Any, Callable, TypeVar

import config

IST = pytz.timezone("Asia/Kolkata")
F = TypeVar("F", bound=Callable[..., Any])


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to both console and the central log file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(config.LOG_PATH)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def ist_time_str() -> str:
    return now_ist().strftime("%H:%M:%S")


def is_market_open() -> bool:
    """True only during NSE intraday session 09:15–15:20 IST on weekdays."""
    dt = now_ist()
    if dt.weekday() >= 5:
        return False
    t = dt.time()
    from datetime import time as dtime
    return dtime(9, 15) <= t <= dtime(15, 20)


def market_opens_in_seconds() -> float:
    """Seconds until next market open (09:15 IST)."""
    dt = now_ist()
    from datetime import time as dtime
    open_today = dt.replace(hour=9, minute=15, second=0, microsecond=0)
    if dt >= open_today:
        open_today = open_today.replace(day=dt.day + 1)
    return (open_today - dt).total_seconds()


def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Decorator: retry on exception with exponential backoff."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait = delay
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    logging.getLogger("retry").warning(
                        f"{fn.__name__} failed ({exc}), retry {attempt}/{max_attempts} in {wait}s"
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper  # type: ignore
    return decorator


def round_to_tick(price: float, tick: float = 0.05) -> float:
    """Round price to nearest NSE tick size."""
    return round(round(price / tick) * tick, 2)


def calc_quantity(risk_amount: float, stop_distance: float, price: float) -> int:
    """Position size: min of risk-based qty and 20% capital limit."""
    if stop_distance <= 0:
        return 0
    risk_qty = int(risk_amount / stop_distance)
    capital_qty = int((config.CAPITAL * 0.20) / price)
    return max(1, min(risk_qty, capital_qty))


def est_charges(trade_value: float) -> float:
    """Rough charge estimate for a round-trip trade."""
    brokerage = 40.0
    stt = trade_value * 0.00025
    nse_txn = trade_value * 2 * 0.0000297
    stamp = trade_value * 0.00003
    gst = (brokerage + nse_txn) * 0.18
    sebi = trade_value * 2 / 1e7
    return round(brokerage + stt + nse_txn + stamp + gst + sebi, 2)


def net_pnl(gross: float, num_trades: int) -> float:
    return round(gross - num_trades * config.CHARGES_PER_TRADE, 2)


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return round((new - old) / abs(old) * 100, 2)
