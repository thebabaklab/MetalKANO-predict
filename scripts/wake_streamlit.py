"""Wake the Streamlit Community Cloud app by simulating a real browser visit.

A plain HTTP GET (curl, UptimeRobot) does NOT wake a sleeping Streamlit app —
the "Yes, get this app back up!" button is rendered by JS, and the app only
counts a visit as activity when a websocket session is established.

This script uses Playwright + headless Chromium to:
  1. Load the app URL like a real browser.
  2. If the wake button is visible, click it and wait for the app to boot.
  3. Otherwise just keep the page open long enough to register a session.
"""

from __future__ import annotations

import datetime
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = "https://metalkano-predict.streamlit.app/"
# Case-insensitive regex — Playwright's name= accepts str or compiled Pattern,
# NOT a lambda. Matches the Streamlit Community Cloud wake prompt.
WAKE_TEXT_RE = re.compile(r"get this app back up", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"{datetime.datetime.utcnow().isoformat()}Z  {msg}", flush=True)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        )
        page = context.new_page()
        log(f"navigating to {APP_URL}")
        try:
            page.goto(APP_URL, wait_until="networkidle", timeout=60_000)
        except PlaywrightTimeoutError:
            log("initial networkidle timed out; continuing anyway")

        # Give React a moment to render either the app or the sleep splash.
        page.wait_for_timeout(3_000)

        try:
            # Try button role first, then any element containing the text —
            # Streamlit Cloud has rendered this control as both <button> and <a>.
            candidates = [
                page.get_by_role("button", name=WAKE_TEXT_RE),
                page.get_by_text(WAKE_TEXT_RE),
            ]
            wake_button = None
            for loc in candidates:
                if loc.count() > 0 and loc.first.is_visible():
                    wake_button = loc.first
                    break

            if wake_button is not None:
                log("sleep splash detected — clicking wake button")
                wake_button.click()
                # Wait for the app to finish booting. This can take a while.
                try:
                    page.wait_for_load_state("networkidle", timeout=180_000)
                except PlaywrightTimeoutError:
                    log("post-wake networkidle timed out; app may still be booting")
                # Hold the session open a bit so Streamlit registers it.
                page.wait_for_timeout(10_000)
                log("wake click complete")
            else:
                log("no sleep splash — app already awake")
                # Hold the session open briefly to make sure the websocket
                # connects and the visit is counted.
                page.wait_for_timeout(5_000)
        finally:
            context.close()
            browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
