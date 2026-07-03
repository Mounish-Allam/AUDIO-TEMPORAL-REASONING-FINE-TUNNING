"""Regenerate assets/space_interface.png -- a screenshot of hf_space/app.py
for the README. Launches the Space app locally (CPU-only, no model loading)
and screenshots it with Playwright.

Deliberately does NOT screenshot the live huggingface.co page: HF injects a
small chrome overlay (repo name + like button) on top of the app that
overlaps the page title. Screenshotting the app run locally avoids that and
produces the exact same content, since the Space runs the same app.py.

Requires: pip install playwright && playwright install chromium
"""
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hf_space"))
from app import CSS, THEME, demo  # noqa: E402

PORT = 7866
OUT = ROOT / "assets" / "space_interface.png"

server = threading.Thread(
    target=lambda: demo.launch(
        server_name="127.0.0.1", server_port=PORT, share=False,
        prevent_thread_lock=False, theme=THEME, css=CSS, show_error=True,
    ),
    daemon=True,
)
server.start()
time.sleep(4)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(f"http://127.0.0.1:{PORT}/", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_selector("text=Results Showcase", timeout=15000)
    page.wait_for_selector("text=qlora_v1", timeout=15000)
    page.wait_for_selector("img[src*='.png'], canvas", timeout=15000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT), full_page=True)
    browser.close()

print(f"Saved {OUT}")
sys.exit(0)  # daemon server thread dies with the process
