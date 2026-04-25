"""Upstox order execution layer — wraps upstox-python-sdk."""

import time
from typing import Optional

import upstox_client
from upstox_client.rest import ApiException

import config
from auth.upstox_auth import UpstoxAuth
from database.trade_logger import TradeLogger
from utils.helpers import setup_logger, round_to_tick, retry

logger = setup_logger("upstox_trader")


class UpstoxTrader:
    """Place, manage, and close Upstox intraday orders."""

    def __init__(
        self,
        auth: Optional[UpstoxAuth] = None,
        trade_logger: Optional[TradeLogger] = None,
    ) -> None:
        self._auth = auth or UpstoxAuth()
        self._logger = trade_logger or TradeLogger()
        self._open_trade_ids: dict[str, int] = {}  # symbol → db trade_id
        self._open_sl_orders: dict[str, str] = {}  # symbol → sl_order_id

    # ── Public interface ──────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=2.0)
    def place_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float,
        sl: float,
        target: float,
        signal_score: float = 0.0,
        regime: str = "",
    ) -> Optional[str]:
        """Place entry order + immediate SL bracket. Returns order_id or None."""
        token = config.INSTRUMENT_MAP.get(symbol)
        if not token:
            logger.error(f"No instrument token for {symbol}")
            return None

        if config.PAPER_MODE:
            return self._place_paper_order(
                symbol, action, qty, price, sl, target, signal_score, regime
            )

        api = self._order_api()

        # ── Entry order ───────────────────────────────────────────────────────
        entry_req = upstox_client.PlaceOrderV3Request(
            quantity=qty,
            product="I",
            validity="DAY",
            price=round_to_tick(price),
            instrument_token=token,
            order_type="LIMIT",
            transaction_type=action.upper(),
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
            slice=True,
        )
        try:
            entry_resp = api.place_order(entry_req)
            order_id = entry_resp.data.order_id
            logger.info(f"[LIVE] Entry order placed: {symbol} {action} {qty}@{price} | ID={order_id}")
        except ApiException as exc:
            logger.error(f"Entry order failed for {symbol}: {exc}")
            return None

        # ── SL order ──────────────────────────────────────────────────────────
        sl_order_id = self._place_sl_order(api, symbol, token, action, qty, sl)

        # ── Log to DB ─────────────────────────────────────────────────────────
        trade_id = self._logger.log_entry(
            symbol=symbol,
            action=action,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            target=target,
            upstox_order_id=order_id,
            regime=regime,
            signal_score=signal_score,
            paper=False,
        )
        self._open_trade_ids[symbol] = trade_id
        self._open_sl_orders[symbol] = sl_order_id or ""
        return order_id

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close an open intraday position with a MARKET order."""
        token = config.INSTRUMENT_MAP.get(symbol)
        if not token:
            return None

        if config.PAPER_MODE:
            return self._close_paper_position(symbol)

        positions = self.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol and p["qty"] != 0), None)
        if not pos:
            logger.warning(f"No open position found for {symbol}")
            return None

        api = self._order_api()
        close_action = "SELL" if pos["qty"] > 0 else "BUY"
        close_req = upstox_client.PlaceOrderV3Request(
            quantity=abs(pos["qty"]),
            product="I",
            validity="DAY",
            price=0,
            instrument_token=token,
            order_type="MARKET",
            transaction_type=close_action,
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
            slice=False,
        )
        try:
            resp = api.place_order(close_req)
            close_order_id = resp.data.order_id
        except ApiException as exc:
            logger.error(f"Close order failed for {symbol}: {exc}")
            return None

        # Cancel pending SL
        if symbol in self._open_sl_orders and self._open_sl_orders[symbol]:
            self.cancel_order(self._open_sl_orders[symbol])

        # Update DB
        trade_id = self._open_trade_ids.get(symbol)
        if trade_id:
            self._logger.log_exit(
                trade_id=trade_id,
                exit_price=pos["ltp"],
                status="CLOSED",
                sl_order_id=close_order_id,
            )
            del self._open_trade_ids[symbol]

        logger.info(f"[LIVE] Closed {symbol} @ {pos['ltp']}")
        return {"symbol": symbol, "exit_price": pos["ltp"], "order_id": close_order_id}

    def close_all_positions(self) -> list[dict]:
        """Close every open intraday position."""
        positions = self.get_positions()
        closed = []
        for pos in positions:
            if pos.get("qty", 0) != 0:
                result = self.close_position(pos["symbol"])
                if result:
                    closed.append(result)
        return closed

    @retry(max_attempts=3, delay=1.0)
    def get_positions(self) -> list[dict]:
        if config.PAPER_MODE:
            return self._paper_positions()

        cfg = self._auth.configure_upstox_client()
        api = upstox_client.PortfolioApi(upstox_client.ApiClient(cfg))
        try:
            resp = api.get_positions()
            positions = []
            for p in (resp.data or []):
                sym = self._token_to_symbol(p.instrument_token)
                positions.append({
                    "symbol": sym,
                    "qty": p.quantity,
                    "avg_price": p.average_price,
                    "ltp": p.last_price,
                    "pnl": p.pnl,
                    "status": "OPEN" if p.quantity != 0 else "CLOSED",
                    "instrument_token": p.instrument_token,
                })
            return positions
        except ApiException as exc:
            logger.error(f"get_positions failed: {exc}")
            return []

    @retry(max_attempts=3, delay=1.0)
    def get_order_book(self) -> list[dict]:
        if config.PAPER_MODE:
            return []
        cfg = self._auth.configure_upstox_client()
        api = upstox_client.OrderApi(upstox_client.ApiClient(cfg))
        try:
            resp = api.get_order_book()
            return [vars(o) for o in (resp.data or [])]
        except ApiException as exc:
            logger.error(f"get_order_book failed: {exc}")
            return []

    @retry(max_attempts=2, delay=1.0)
    def get_funds(self) -> dict:
        if config.PAPER_MODE:
            return {"available": config.CAPITAL, "used": 0.0}
        cfg = self._auth.configure_upstox_client()
        api = upstox_client.UserApi(upstox_client.ApiClient(cfg))
        try:
            resp = api.get_user_fund_margin(segment="SEC")
            d = resp.data
            return {
                "available": d.equity.available_margin,
                "used": d.equity.used_margin,
                "total": d.equity.total_collateral,
            }
        except ApiException as exc:
            logger.error(f"get_funds failed: {exc}")
            return {"available": 0, "used": 0}

    def cancel_order(self, order_id: str) -> bool:
        if config.PAPER_MODE:
            logger.info(f"[PAPER] Cancel order {order_id}")
            return True
        cfg = self._auth.configure_upstox_client()
        api = upstox_client.OrderApi(upstox_client.ApiClient(cfg))
        try:
            api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except ApiException as exc:
            logger.error(f"cancel_order failed: {exc}")
            return False

    # ── Paper trading simulation ──────────────────────────────────────────────

    def _place_paper_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float,
        sl: float,
        target: float,
        signal_score: float,
        regime: str,
    ) -> str:
        fake_order_id = f"PAPER_{symbol}_{int(time.time())}"
        trade_id = self._logger.log_entry(
            symbol=symbol,
            action=action,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            target=target,
            upstox_order_id=fake_order_id,
            regime=regime,
            signal_score=signal_score,
            paper=True,
        )
        self._open_trade_ids[symbol] = trade_id
        logger.info(
            f"[PAPER] Entry: {symbol} {action} {qty}@{price} | "
            f"SL={sl} | Target={target} | ID={fake_order_id}"
        )
        return fake_order_id

    def _close_paper_position(self, symbol: str) -> Optional[dict]:
        trade_id = self._open_trade_ids.get(symbol)
        if not trade_id:
            return None
        # Simulate exit at current price (use 0 as placeholder — real price injected by caller)
        from data.market_data import MarketData
        # Attempt to get live price; fall back to entry
        ltp = 0.0
        try:
            ltp = MarketData().get_live_price(symbol)
        except Exception:
            pass
        self._logger.log_exit(trade_id=trade_id, exit_price=ltp or 0.0, status="CLOSED")
        del self._open_trade_ids[symbol]
        logger.info(f"[PAPER] Closed {symbol} @ {ltp}")
        return {"symbol": symbol, "exit_price": ltp}

    def _paper_positions(self) -> list[dict]:
        open_trades = self._logger.get_open_trades()
        return [
            {
                "symbol": t["symbol"],
                "qty": t["quantity"],
                "avg_price": t["entry_price"],
                "ltp": t["entry_price"],  # placeholder
                "pnl": 0.0,
                "status": "OPEN",
            }
            for t in open_trades
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _order_api(self) -> upstox_client.OrderApiV3:
        cfg = self._auth.configure_upstox_client()
        return upstox_client.OrderApiV3(upstox_client.ApiClient(cfg))

    def _place_sl_order(
        self,
        api: upstox_client.OrderApiV3,
        symbol: str,
        token: str,
        action: str,
        qty: int,
        sl: float,
    ) -> Optional[str]:
        sl_action = "SELL" if action.upper() == "BUY" else "BUY"
        sl_req = upstox_client.PlaceOrderV3Request(
            quantity=qty,
            product="I",
            validity="DAY",
            price=round_to_tick(sl - 0.5 if sl_action == "SELL" else sl + 0.5),
            instrument_token=token,
            order_type="SL",
            transaction_type=sl_action,
            disclosed_quantity=0,
            trigger_price=round_to_tick(sl),
            is_amo=False,
            slice=False,
        )
        try:
            resp = api.place_order(sl_req)
            sl_id = resp.data.order_id
            logger.info(f"SL order placed for {symbol}: trigger={sl} | ID={sl_id}")
            return sl_id
        except ApiException as exc:
            logger.warning(f"SL order failed for {symbol}: {exc}")
            return None

    def _token_to_symbol(self, token: str) -> str:
        for sym, tok in config.INSTRUMENT_MAP.items():
            if tok == token:
                return sym
        return token
