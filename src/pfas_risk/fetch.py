"""Download MassDEP's bulk waste site data, which mass.gov only serves to browsers.

mass.gov answers scripted requests with HTTP 403, so a plain download is tried first and a
headless Chromium (Playwright, the ``refresh`` extra) is the fallback. The file's sha256 is
compared with the one recorded by the last ``pfas-risk releases`` build, so scheduled runs
can stop early when MassDEP hasn't published anything new.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .config import DATA_DIR
from .sources import local_path, source

log = logging.getLogger(__name__)

LISTING_URL = "https://www.mass.gov/info-details/downloadable-contaminated-site-lists"
SOURCE_RECORD = DATA_DIR / "releases" / "massdep_source.json"
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/140.0 Safari/537.36")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except urllib.error.HTTPError as e:
        log.info("plain download refused (HTTP %s); using a headless browser", e.code)
        return False


def _browser_download(url: str, dest: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("mass.gov refused a plain download; install the browser fallback with "
                           "`pip install -e '.[refresh]' && python -m playwright install chromium`") from e
    with sync_playwright() as p:
        # PFAS_RISK_CHROMIUM points at an existing Chromium when Playwright's own isn't installed.
        browser = p.chromium.launch(executable_path=os.environ.get("PFAS_RISK_CHROMIUM") or None)
        try:
            page = browser.new_context(accept_downloads=True).new_page()
            page.goto(LISTING_URL, timeout=120_000)  # pick up the cookies the download expects
            with page.expect_download(timeout=300_000) as dl:
                page.evaluate("url => { window.location.href = url; }", url)
            dl.value.save_as(dest)
        finally:
            browser.close()


def fetch_release_zip(dest: Path | None = None) -> Path:
    """Download the MassDEP zip to data/raw (atomically) and return its path."""
    dest = Path(dest or local_path("massdep_releases"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = source("massdep_releases")["url"]
    if not _plain_download(url, tmp):
        _browser_download(url, tmp)
    with open(tmp, "rb") as f:
        if f.read(2) != b"PK":
            tmp.unlink()
            raise RuntimeError(f"{url} did not return a zip file")
    tmp.replace(dest)
    return dest


def recorded_sha256() -> str | None:
    if SOURCE_RECORD.exists():
        return json.loads(SOURCE_RECORD.read_text()).get("zip_sha256")
    return None


def record_source(zip_path: Path, n_releases: int) -> None:
    """Called by the release-list build: remember which download it came from."""
    import zipfile
    from datetime import datetime, timezone

    with zipfile.ZipFile(zip_path) as z:
        stamp = max(i.date_time for i in z.infolist())
    SOURCE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_RECORD.write_text(json.dumps({
        "zip_sha256": sha256(zip_path),
        "tables_dated": f"{stamp[0]:04d}-{stamp[1]:02d}-{stamp[2]:02d}",
        "built": datetime.now(timezone.utc).date().isoformat(),
        "pfas_releases": n_releases,
    }, indent=2) + "\n")
