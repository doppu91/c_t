"""Upstox OAuth 2.0 authentication with daily TOTP-based auto-refresh."""

import asyncio
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import pyotp
import requests
import upstox_client

import config
from utils.helpers import setup_logger, retry

logger = setup_logger("upstox_auth")

_ENV_FILE = Path(__file__).parent.parent / ".env"


class UpstoxAuth:
    """Handles Upstox OAuth token lifecycle with TOTP auto-login."""

    LOGIN_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"

    # JWT exp for the provided extended token: 1808431200 (~June 2027)
    _EXTENDED_TOKEN_EXPIRY = 1808431200

    def __init__(self) -> None:
        self._access_token: str = config.UPSTOX_ACCESS_TOKEN
        # If a token is already present, honour its expiry from the JWT
        if self._access_token:
            self._token_expiry: float = self._parse_jwt_expiry(self._access_token)
            logger.info(
                f"Loaded existing token. Valid until "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(self._token_expiry))}"
            )
        else:
            self._token_expiry: float = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def is_token_valid(self) -> bool:
        return bool(self._access_token) and time.time() < self._token_expiry

    def get_token(self) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        return self._access_token

    def configure_upstox_client(self) -> upstox_client.Configuration:
        """Return a fully configured upstox_client.Configuration."""
        cfg = upstox_client.Configuration()
        cfg.access_token = self.get_token()
        if config.PAPER_MODE:
            cfg.host = "https://sandbox-api.upstox.com"
        return cfg

    def refresh_token(self) -> bool:
        """Attempt TOTP auto-login; falls back to manual alert on failure."""
        logger.info("Starting daily token refresh via TOTP...")
        try:
            auth_code = self._get_auth_code_via_playwright()
            token = self._exchange_auth_code(auth_code)
            self._save_token(token)
            logger.info("Token refreshed successfully.")
            self._notify("✅ Upstox token refreshed successfully at %s" % time.strftime("%H:%M IST"))
            return True
        except Exception as exc:
            logger.error(f"TOTP auto-login failed: {exc}")
            self._notify(
                "⚠️ Upstox manual token refresh required!\n"
                f"Error: {exc}\n"
                "Trading paused until token is provided."
            )
            return False

    def test(self) -> None:
        """Quick connectivity test — logs fund info."""
        cfg = self.configure_upstox_client()
        api = upstox_client.UserApi(upstox_client.ApiClient(cfg))
        funds = api.get_user_fund_margin(segment="SEC")
        logger.info(f"Connection OK. Funds: {funds}")
        print("Connection test passed.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_auth_code_via_playwright(self) -> str:
        """Use headless Chromium + TOTP to capture the OAuth auth code."""
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        totp_code = pyotp.TOTP(config.UPSTOX_TOTP_SECRET).now()

        login_url = (
            f"{self.LOGIN_URL}"
            f"?response_type=code"
            f"&client_id={config.UPSTOX_API_KEY}"
            f"&redirect_uri={urllib.parse.quote(config.UPSTOX_REDIRECT_URI)}"
        )

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            auth_code: Optional[str] = None

            def handle_request(request):
                nonlocal auth_code
                url = request.url
                if config.UPSTOX_REDIRECT_URI in url:
                    parsed = urllib.parse.urlparse(url)
                    params = urllib.parse.parse_qs(parsed.query)
                    if "code" in params:
                        auth_code = params["code"][0]

            page.on("request", handle_request)

            try:
                page.goto(login_url, timeout=30000)
                # Mobile number
                page.wait_for_selector('input[type="text"]', timeout=15000)
                page.fill('input[type="text"]', config.UPSTOX_MOBILE)
                page.click('button[type="submit"]')
                # PIN
                page.wait_for_selector('input[type="password"]', timeout=10000)
                page.fill('input[type="password"]', config.UPSTOX_PIN)
                page.click('button[type="submit"]')
                # TOTP
                page.wait_for_selector('input[placeholder*="OTP"]', timeout=10000)
                page.fill('input[placeholder*="OTP"]', totp_code)
                page.click('button[type="submit"]')
                # Wait for redirect — google.com redirect, code in URL params
                page.wait_for_url("https://www.google.com*", timeout=15000)
            except PWTimeout:
                try:
                    page.fill('input[name="otp"]', totp_code)
                    page.click('button[type="submit"]')
                    page.wait_for_url("https://www.google.com*", timeout=10000)
                except Exception:
                    pass

            # Capture auth code from final URL if handler missed it
            if not auth_code:
                final_url = page.url
                parsed = urllib.parse.urlparse(final_url)
                params = urllib.parse.parse_qs(parsed.query)
                auth_code = params.get("code", [None])[0]

            browser.close()

        if not auth_code:
            raise RuntimeError("Could not capture auth code from redirect URL")
        return auth_code

    @retry(max_attempts=3, delay=2.0)
    def _exchange_auth_code(self, auth_code: str) -> str:
        """Exchange auth code for access token via Upstox API."""
        resp = requests.post(
            self.TOKEN_URL,
            data={
                "code": auth_code,
                "client_id": config.UPSTOX_API_KEY,
                "client_secret": config.UPSTOX_API_SECRET,
                "redirect_uri": config.UPSTOX_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token", "")
        if not token:
            raise ValueError(f"No access_token in response: {data}")
        return token

    def _save_token(self, token: str) -> None:
        """Persist access token to .env file and update in-memory state."""
        self._access_token = token
        # Token expires next day at 06:00 IST (~15.5 hours from 8 AM)
        self._token_expiry = time.time() + 55_800

        if _ENV_FILE.exists():
            content = _ENV_FILE.read_text()
            if "UPSTOX_ACCESS_TOKEN=" in content:
                content = re.sub(
                    r"^UPSTOX_ACCESS_TOKEN=.*$",
                    f"UPSTOX_ACCESS_TOKEN={token}",
                    content,
                    flags=re.MULTILINE,
                )
            else:
                content += f"\nUPSTOX_ACCESS_TOKEN={token}\n"
            _ENV_FILE.write_text(content)

        # Also update runtime env
        os.environ["UPSTOX_ACCESS_TOKEN"] = token
        config.UPSTOX_ACCESS_TOKEN = token

    @staticmethod
    def _parse_jwt_expiry(token: str) -> float:
        """Extract exp claim from JWT without verifying signature."""
        try:
            import base64, json as _json
            parts = token.split(".")
            if len(parts) < 2:
                return 0.0
            payload = parts[1]
            # Add padding
            payload += "=" * (4 - len(payload) % 4)
            data = _json.loads(base64.urlsafe_b64decode(payload))
            return float(data.get("exp", 0))
        except Exception:
            return 0.0

    def _notify(self, message: str) -> None:
        """Fire-and-forget Telegram notification."""
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )
        except Exception as exc:
            logger.warning(f"Telegram notify failed: {exc}")
