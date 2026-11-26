"""Render a recreation page to WAV in headless Chromium (OfflineAudioContext), and
confirm the page made no network requests."""
import base64
from pathlib import Path

import numpy as np
import soundfile as sf


def render_offline(page_path, out_wav, loops=2, sr=44100):
    from playwright.sync_api import sync_playwright
    url = Path(page_path).resolve().as_uri()
    requests = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("request", lambda r: requests.append(r.url) if r.url != url else None)
        page.goto(url)
        b64 = page.evaluate(f"renderOffline({loops}, {sr})")
        browser.close()
    pcm = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    sf.write(out_wav, pcm, sr, subtype="PCM_16")
    return pcm, sr, requests
