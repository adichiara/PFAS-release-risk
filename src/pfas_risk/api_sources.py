"""Public sources served by web APIs rather than as files (see `fetch:` in config/sources.yaml).

Each fetcher writes one local file that `sources.download` then records like any other
download: ArcGIS feature services as GeoJSON, EPA TRI and the MassDEP drinking-water results
as CSV.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TIMEOUT = 180


def _get(url: str, attempts: int = 4) -> bytes:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return r.read()
        except OSError as e:  # URLError, connection resets, timeouts
            if i == attempts - 1:
                raise
            log.info("retrying %s after %s", url, e)
            time.sleep(2 ** (i + 1))
    raise AssertionError("unreachable")


def arcgis(spec: dict, dest: Path) -> None:
    """All features of an ArcGIS layer matching `where`, as GeoJSON (WGS84)."""
    features, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"where": spec["where"], "outFields": spec.get("fields", "*"),
                                    "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000})
        page = json.loads(_get(f"{spec['service']}/query?{q}"))
        if "error" in page:
            raise RuntimeError(f"{spec['service']}: {page['error']}")
        features += page["features"]
        if not page.get("properties", {}).get("exceededTransferLimit") and not page.get("exceededTransferLimit"):
            break
        offset += len(page["features"])
    if not features:
        raise RuntimeError(f"{spec['service']} returned no features for {spec['where']!r}")
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def tri(spec: dict, dest: Path) -> None:
    """EPA Toxics Release Inventory basic data for Massachusetts, several reporting years."""
    frames = []
    for year in range(spec["first_year"], spec["last_year"] + 1):
        url = spec["url"].format(year=year)
        try:
            frames.append(pd.read_csv(io.BytesIO(_get(url)), dtype=str))
        except (OSError, pd.errors.ParserError) as e:
            log.warning("TRI %d unavailable (%s); continuing without it", year, e)
    if not frames:
        raise RuntimeError("no TRI year could be downloaded")
    pd.concat(frames).to_csv(dest, index=False)


def eea_drinking_water(spec: dict, dest: Path) -> None:
    """One analyte's results for every public water system, from the EEA Data Portal API.

    The API returns at most 100 records per query and ignores its paging parameters, so this
    queries each water system (raw and finished water separately). A system with more than
    100 results in one of those is represented by the 100 the API returns.
    """
    from .sources import read  # the water-system list is itself a catalog source

    systems = read(spec["systems_from"])
    systems = systems[~systems["TYPE"].isin(spec.get("skip_types", []))]
    pws_ids = sorted(systems["PWS_ID"].dropna().unique())
    base = spec["url"] + "?" + urllib.parse.urlencode({"ChemicalName": spec["chemical"]})

    def one(args: tuple[str, str]) -> tuple[int, list[dict]]:
        pws, kind = args
        page = json.loads(_get(f"{base}&PWSId={pws}&RaworFinished={kind}"))
        return page["TotalCount"], page["Items"]

    rows, truncated = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for total, items in pool.map(one, [(p, k) for p in pws_ids for k in ("R", "F")]):
            rows += items
            truncated += total > len(items)
    log.info("%s: %d records from %d water systems (%d queries truncated at 100)",
             spec["chemical"], len(rows), len(pws_ids), truncated)
    if not rows:
        raise RuntimeError("drinking-water API returned no records")
    pd.DataFrame(rows).drop_duplicates("Id").to_csv(dest, index=False)


FETCHERS = {"arcgis": arcgis, "tri": tri, "eea_drinking_water": eea_drinking_water}


def fetch(spec: dict, dest: Path) -> str:
    """Run the fetcher named in `spec['kind']`; return a URL describing what was fetched."""
    FETCHERS[spec["kind"]](spec, dest)
    return spec.get("service") or spec["url"]
