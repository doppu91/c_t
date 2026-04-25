"""SQLite trade journal — records every order, position, and daily summary."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Generator, Optional

import config
from utils.helpers import setup_logger, today_ist

logger = setup_logger("trade_logger")

_lock = threading.Lock()


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(config.DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


class TradeLogger:
    """Persistent trade journal backed by SQLite."""

    def __init__(self) -> None:
        self._ensure_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with _conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date        TEXT NOT NULL,
                    symbol            TEXT NOT NULL,
                    action            TEXT NOT NULL,          -- BUY / SELL
                    entry_price       REAL,
                    exit_price        REAL,
                    quantity          INTEGER,
                    stop_loss         REAL,
                    target            REAL,
                    entry_time        TEXT,
                    exit_time         TEXT,
                    status            TEXT DEFAULT 'OPEN',   -- OPEN/CLOSED/SL/TARGET
                    gross_pnl         REAL DEFAULT 0.0,
                    estimated_charges REAL DEFAULT 100.0,
                    net_pnl           REAL DEFAULT 0.0,
                    upstox_order_id   TEXT,
                    sl_order_id       TEXT,
                    regime            TEXT,
                    signal_score      REAL,
                    paper             INTEGER DEFAULT 1,      -- 1 = paper trade
                    notes             TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_summary (
                    summary_date  TEXT PRIMARY KEY,
                    num_trades    INTEGER DEFAULT 0,
                    num_wins      INTEGER DEFAULT 0,
                    num_losses    INTEGER DEFAULT 0,
                    gross_pnl     REAL DEFAULT 0.0,
                    total_charges REAL DEFAULT 0.0,
                    net_pnl       REAL DEFAULT 0.0,
                    regime        TEXT,
                    vix           REAL,
                    notes         TEXT
                );

                CREATE TABLE IF NOT EXISTS signals_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            TEXT NOT NULL,
                    symbol        TEXT NOT NULL,
                    tech_score    REAL,
                    sent_score    REAL,
                    fii_score     REAL,
                    gex_score     REAL,
                    global_score  REAL,
                    combined      REAL,
                    regime        TEXT,
                    action_taken  TEXT
                );
            """)
        logger.info(f"Database ready at {config.DB_PATH}")

    # ── Trade operations ──────────────────────────────────────────────────────

    def log_entry(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target: float,
        upstox_order_id: str = "",
        regime: str = "",
        signal_score: float = 0.0,
        paper: bool = True,
    ) -> int:
        with _lock, _conn() as con:
            cur = con.execute(
                """INSERT INTO trades
                   (trade_date, symbol, action, entry_price, quantity,
                    stop_loss, target, entry_time, upstox_order_id,
                    regime, signal_score, paper, estimated_charges)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(today_ist()),
                    symbol,
                    action,
                    entry_price,
                    quantity,
                    stop_loss,
                    target,
                    datetime.utcnow().isoformat(),
                    upstox_order_id,
                    regime,
                    signal_score,
                    int(paper),
                    config.CHARGES_PER_TRADE,
                ),
            )
            return cur.lastrowid  # type: ignore

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        status: str = "CLOSED",
        sl_order_id: str = "",
    ) -> None:
        with _lock, _conn() as con:
            row = con.execute(
                "SELECT entry_price, quantity FROM trades WHERE id=?", (trade_id,)
            ).fetchone()
            if not row:
                logger.warning(f"Trade id {trade_id} not found for exit logging")
                return
            entry, qty = row["entry_price"], row["quantity"]
            gross = round((exit_price - entry) * qty, 2)
            net = round(gross - config.CHARGES_PER_TRADE, 2)
            con.execute(
                """UPDATE trades SET exit_price=?, exit_time=?, status=?,
                   gross_pnl=?, net_pnl=?, sl_order_id=? WHERE id=?""",
                (
                    exit_price,
                    datetime.utcnow().isoformat(),
                    status,
                    gross,
                    net,
                    sl_order_id,
                    trade_id,
                ),
            )

    def get_open_trades(self, trade_date: Optional[str] = None) -> list[dict]:
        d = trade_date or str(today_ist())
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM trades WHERE trade_date=? AND status='OPEN'", (d,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self, trade_date: Optional[str] = None) -> dict[str, float]:
        d = trade_date or str(today_ist())
        with _conn() as con:
            row = con.execute(
                """SELECT COUNT(*) as n, SUM(gross_pnl) as gross,
                          SUM(estimated_charges) as charges, SUM(net_pnl) as net
                   FROM trades WHERE trade_date=? AND status != 'OPEN'""",
                (d,),
            ).fetchone()
        n = row["n"] or 0
        gross = round(row["gross"] or 0.0, 2)
        charges = round(row["charges"] or 0.0, 2)
        net = round(row["net"] or 0.0, 2)
        return {"num_trades": n, "gross_pnl": gross, "charges": charges, "net_pnl": net}

    # ── Daily summary ─────────────────────────────────────────────────────────

    def upsert_daily_summary(self, regime: str = "", vix: float = 0.0) -> dict:
        d = str(today_ist())
        pnl = self.get_daily_pnl(d)
        wins = self._count_by_status(d, "TARGET") + self._count_positive(d)
        losses = self._count_negative(d)
        with _lock, _conn() as con:
            con.execute(
                """INSERT INTO daily_summary
                   (summary_date, num_trades, num_wins, num_losses,
                    gross_pnl, total_charges, net_pnl, regime, vix)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(summary_date) DO UPDATE SET
                   num_trades=excluded.num_trades,
                   num_wins=excluded.num_wins,
                   num_losses=excluded.num_losses,
                   gross_pnl=excluded.gross_pnl,
                   total_charges=excluded.total_charges,
                   net_pnl=excluded.net_pnl,
                   regime=excluded.regime,
                   vix=excluded.vix""",
                (
                    d,
                    pnl["num_trades"],
                    wins,
                    losses,
                    pnl["gross_pnl"],
                    pnl["charges"],
                    pnl["net_pnl"],
                    regime,
                    vix,
                ),
            )
        return pnl

    def get_weekly_summary(self) -> list[dict]:
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM daily_summary ORDER BY summary_date DESC LIMIT 7"
            ).fetchall()
        return [dict(r) for r in rows]

    def log_signal(self, symbol: str, scores: dict, regime: str, action: str) -> None:
        with _lock, _conn() as con:
            con.execute(
                """INSERT INTO signals_log
                   (ts,symbol,tech_score,sent_score,fii_score,
                    gex_score,global_score,combined,regime,action_taken)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    scores.get("technical", 0),
                    scores.get("sentiment", 0),
                    scores.get("fii_flow", 0),
                    scores.get("gex", 0),
                    scores.get("global_cue", 0),
                    scores.get("combined", 0),
                    regime,
                    action,
                ),
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _count_by_status(self, d: str, status: str) -> int:
        with _conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM trades WHERE trade_date=? AND status=?",
                (d, status),
            ).fetchone()[0]

    def _count_positive(self, d: str) -> int:
        with _conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM trades WHERE trade_date=? AND gross_pnl > 0",
                (d,),
            ).fetchone()[0]

    def _count_negative(self, d: str) -> int:
        with _conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM trades WHERE trade_date=? AND gross_pnl < 0",
                (d,),
            ).fetchone()[0]
