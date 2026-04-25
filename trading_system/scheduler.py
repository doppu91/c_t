"""APScheduler-based task scheduler — all jobs run in IST."""

import logging
from datetime import datetime
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.helpers import setup_logger

logger = setup_logger("scheduler")
IST = pytz.timezone("Asia/Kolkata")


class TradingScheduler:
    """Schedules all recurring trading system tasks."""

    def __init__(
        self,
        auth,
        market_data,
        regime_hmm,
        lgbm_trainer,
        global_cues,
        signal_combiner,
        trader,
        risk_manager,
        trade_logger,
        telegram_send,
        morning_brief_fn,
        eod_fn,
        scan_fn,
    ) -> None:
        self._auth = auth
        self._market_data = market_data
        self._regime_hmm = regime_hmm
        self._lgbm_trainer = lgbm_trainer
        self._global_cues = global_cues
        self._combiner = signal_combiner
        self._trader = trader
        self._risk_manager = risk_manager
        self._trade_logger = trade_logger
        self._send = telegram_send
        self._morning_brief = morning_brief_fn
        self._eod = eod_fn
        self._scan = scan_fn

        self._scheduler = BackgroundScheduler(timezone=IST)
        self._current_regime: str = "Sideways"
        self._vix: float = 15.0

    def start(self) -> None:
        self._register_jobs()
        self._scheduler.start()
        logger.info("Scheduler started with IST timezone")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    # ── Job registration ──────────────────────────────────────────────────────

    def _register_jobs(self) -> None:
        s = self._scheduler

        # Token refresh at 8:00 AM Mon-Fri
        s.add_job(
            self._job_token_refresh,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=0, timezone=IST),
            id="token_refresh",
            name="Daily token refresh",
            misfire_grace_time=300,
        )

        # Global cues at 8:30 AM
        s.add_job(
            self._job_global_scan,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone=IST),
            id="global_scan",
            name="Global cues scan",
        )

        # Regime detection at 8:45 AM
        s.add_job(
            self._job_regime_check,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=45, timezone=IST),
            id="regime_check",
            name="Regime detection",
        )

        # Morning brief at 8:59 AM
        s.add_job(
            self._job_morning_brief,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=59, timezone=IST),
            id="morning_brief",
            name="Morning brief",
        )

        # 5-min scan loop during market hours
        s.add_job(
            self._job_scan,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/5",
                timezone=IST,
            ),
            id="market_scan",
            name="5-min signal scan",
        )

        # Market close — close all positions at 15:20
        s.add_job(
            self._job_market_close,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=20, timezone=IST),
            id="market_close",
            name="Force close all positions",
        )

        # EOD summary at 15:30
        s.add_job(
            self._job_eod_summary,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone=IST),
            id="eod_summary",
            name="EOD summary",
        )

        # Weekly model retrain on Sunday at 23:00
        s.add_job(
            self._job_weekly_retrain,
            CronTrigger(day_of_week="sun", hour=23, minute=0, timezone=IST),
            id="weekly_retrain",
            name="Weekly model retrain",
        )

        logger.info(f"Registered {len(s.get_jobs())} scheduled jobs")

    # ── Job implementations ───────────────────────────────────────────────────

    def _job_token_refresh(self) -> None:
        logger.info("Job: daily token refresh")
        success = self._auth.refresh_token()
        if not success:
            self._send("⚠️ Token refresh failed — trading paused until manual token provided")

    def _job_global_scan(self) -> None:
        logger.info("Job: global cues scan")
        try:
            cues = self._global_cues.fetch()
            self._vix = cues.get("vix_value", 15.0)
            logger.info(f"Global cues fetched: score={cues.get('score')}, VIX={self._vix:.1f}")
        except Exception as exc:
            logger.error(f"Global scan error: {exc}")

    def _job_regime_check(self) -> None:
        logger.info("Job: regime detection")
        try:
            df = self._market_data.get_candles("NIFTY_INDEX", timeframe="1d", bars=250)
            if df.empty:
                import yfinance as yf
                raw = yf.download("^NSEI", period="1y", interval="1d", progress=False)
                raw.columns = [c.lower() for c in raw.columns]
                df = raw[["open", "high", "low", "close", "volume"]].dropna()
            self._current_regime = self._regime_hmm.predict(df)
            logger.info(f"Regime detected: {self._current_regime}")
        except Exception as exc:
            logger.error(f"Regime check error: {exc}")

    def _job_morning_brief(self) -> None:
        logger.info("Job: morning brief")
        try:
            self._morning_brief(self._current_regime, self._vix)
        except Exception as exc:
            logger.error(f"Morning brief error: {exc}")

    def _job_scan(self) -> None:
        from utils.helpers import is_market_open
        if not is_market_open():
            return
        try:
            self._scan(self._current_regime, self._vix)
        except Exception as exc:
            logger.error(f"Scan error: {exc}")

    def _job_market_close(self) -> None:
        logger.info("Job: force-close all intraday positions")
        try:
            closed = self._trader.close_all_positions()
            if closed:
                self._send(f"🔒 Auto-closed {len(closed)} position(s) at 15:20 IST")
        except Exception as exc:
            logger.error(f"Market close error: {exc}")

    def _job_eod_summary(self) -> None:
        logger.info("Job: EOD summary")
        try:
            self._eod(self._current_regime, self._vix)
        except Exception as exc:
            logger.error(f"EOD summary error: {exc}")

    def _job_weekly_retrain(self) -> None:
        logger.info("Job: weekly model retrain")
        try:
            self._send("🔄 Starting weekly model retrain...")
            self._lgbm_trainer.train_all()
            self._regime_hmm.train()
            self._send("✅ Weekly model retrain complete")
        except Exception as exc:
            logger.error(f"Retrain error: {exc}")
            self._send(f"❌ Weekly retrain failed: {exc}")

    def get_regime(self) -> str:
        return self._current_regime

    def get_vix(self) -> float:
        return self._vix
