"""Download and read the public datasets listed in config/sources.yaml."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd

from .config import CRS, RAW_DIR, catalog, source

log = logging.getLogger(__name__)

MANIFEST = RAW_DIR / "manifest.json"


def local_path(name: str) -> Path:
    meta = source(name)
    return RAW_DIR / meta.get("file", Path(meta.get("path", name)).name)


def download(name: str, force: bool = False) -> Path:
    """Fetch one `available` source into data/raw, recording its sha256."""
    meta = source(name)
    if meta["status"] != "available":
        raise RuntimeError(f"{name} is '{meta['status']}', not downloadable: {meta.get('notes', '')}")
    dest = local_path(name)
    if dest.exists() and not force:
        return dest
    url = f"{catalog()['massgis_base']}/{meta['path']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    _record(name, url, dest)
    return dest


def download_all(force: bool = False) -> list[Path]:
    names = [n for n, m in catalog()["sources"].items() if m["status"] == "available"]
    return [download(n, force=force) for n in names]


def _record(name: str, url: str, path: Path) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = {"url": url, "file": path.name, "sha256": h.hexdigest(), "bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def read(name: str, **kwargs) -> gpd.GeoDataFrame:
    """Read a vector source (downloading it if needed), in the project CRS."""
    meta = source(name)
    path = download(name)
    if "gdb" in meta:  # file geodatabase: open the .gdb, then pick the layer
        uri = f"zip://{path}!{meta['gdb']}"
        kwargs.setdefault("layer", meta["layer"])
    else:
        uri = f"zip://{path}!{meta['layer']}"
    gdf = gpd.read_file(uri, engine="pyogrio", **kwargs)
    return gdf.to_crs(CRS) if gdf.crs is not None and gdf.crs != CRS else gdf
