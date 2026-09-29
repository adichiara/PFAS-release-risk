"""The response: MassDEP reportable releases (RTNs) involving PFAS.

``pfas-risk releases`` builds data/releases/massdep_pfas_releases.csv from MassDEP's bulk
"Waste Site Cleanup Notifications & Status" download (RELEASE.DBF + CHEMICAL.DBF), adds
sites from the 2021 list that the current database no longer tags with a PFAS chemical,
and attaches MassDEP's own site coordinates from its data portal API. The CSV is small,
public and committed, so the rest of the pipeline does not need the download.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import CRS, DATA_DIR, SEED_RELEASES, WGS84
from .fetch import record_source
from .geocode import geocode, normalize_town
from .sources import local_path, read

log = logging.getLogger(__name__)

RELEASE_LIST = DATA_DIR / "releases" / "massdep_pfas_releases.csv"
SITE_API = "https://eeaonline.eea.state.ma.us/EEA/DataLake/V1.0/DataLakeAPI/searchablesite/"

# MassDEP chemical names are free text: "PFAS6 (SUMMATION)", "PERFLUROHEXANSULFONIC ACID PFHXS",
# "TOTAL REGULATED PFAS COMPOUNDS (PPT)", ... Match the family, including common misspellings.
PFAS_PATTERN = re.compile(
    r"PFAS|PFOS|PFOA|PFHX|PFNA|PFDA|PFBS|PFHPA|GENX|HFPO|PERFLU|POLYFLU|FLUOROALKYL|FLUOROTELOMER",
    re.IGNORECASE,
)

# Column names differ between the seed list, portal exports and our own release list.
_COLUMN_ALIASES = {
    "rtn": ["rtn", "release tracking number"],
    "town": ["town", "city", "city/town", "municipality", "release town"],
    "address": ["address", "release address", "location address", "street address"],
    "site_name": ["site_name", "site name", "site name location aid", "location name", "name"],
    "notification_date": ["notification_date", "notification date", "notif_date", "date notified"],
    "chemical": ["chemical", "chemicals", "chemical name", "oil or hazardous material"],
}


def is_pfas(chemical: str | None) -> bool:
    return isinstance(chemical, str) and bool(PFAS_PATTERN.search(chemical))


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for std, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = std
                break
    missing = {"rtn", "town", "address", "chemical"} - set(rename.values())
    if missing:
        raise ValueError(f"release file is missing columns {sorted(missing)}; found {list(df.columns)}")
    return df.rename(columns=rename)


def _join_chemicals(s: pd.Series) -> str:
    return "; ".join(sorted({c.strip() for c in s.dropna()}))


def pfas_rtns(release: pd.DataFrame, chemical: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    """PFAS releases from the bulk tables, plus seed RTNs the database no longer tags as PFAS."""
    pf = chemical[chemical["CHEMICAL"].map(is_pfas)]
    chems = pf.groupby("RTN")["CHEMICAL"].agg(_join_chemicals)
    rel = release.drop_duplicates("RTN").set_index("RTN")  # the table repeats some rows verbatim
    current = pd.DataFrame({
        "rtn": chems.index,
        "town": rel["TOWN"].reindex(chems.index).values,
        "address": rel["ADDRESS"].reindex(chems.index).values,
        "site_name": rel["SITE_NAME"].reindex(chems.index).values,
        "notification_date": rel["OFC_NOTIF"].reindex(chems.index).values,
        "chemical": chems.values,
        "status": rel["CURRENT_ST"].reindex(chems.index).values,
        "primary_rtn": rel["PRIM_ID"].reindex(chems.index).values,
        "source": "MassDEP database",
    })
    extra = seed[~seed["rtn"].isin(current["rtn"])].copy()
    extra["status"] = extra["rtn"].map(rel["CURRENT_ST"])
    extra["primary_rtn"] = extra["rtn"].map(rel["PRIM_ID"])
    extra["source"] = "2021 list only"
    out = pd.concat([current, extra[current.columns]], ignore_index=True)
    out["notification_date"] = pd.to_datetime(out["notification_date"], format="mixed", errors="coerce").dt.date
    # A release closed and linked into another RTN on this list is the same site; keep the primary.
    linked = out["primary_rtn"].notna() & (out["primary_rtn"] != out["rtn"]) & out["primary_rtn"].isin(out["rtn"])
    return out[~linked].sort_values("rtn").reset_index(drop=True)


def fetch_coordinates(rtns: list[str], pause: float = 0.2) -> pd.DataFrame:
    """MassDEP's own site coordinates (WGS84) from the data portal API, one request per RTN."""
    rows = []
    for rtn in rtns:
        lat = lon = None
        try:
            with urllib.request.urlopen(SITE_API + rtn, timeout=30) as r:
                d = json.load(r)
            lat, lon = d.get("Latitude") or None, d.get("Longitude") or None
        except Exception as e:  # noqa: BLE001 - a missing site just stays unplaced
            log.warning("no coordinates for %s: %s", rtn, e)
        rows.append({"rtn": rtn, "lat": lat, "lon": lon})
        time.sleep(pause)
    out = pd.DataFrame(rows, columns=["rtn", "lat", "lon"])
    out[["lat", "lon"]] = out[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    return out


def build_release_list(zip_path: Path | None = None, coordinates: bool = True,
                       refresh_coordinates: bool = False) -> Path:
    zip_path = zip_path or local_path("massdep_releases")
    if not Path(zip_path).exists():
        raise FileNotFoundError(f"{zip_path} not found; see massdep_releases in config/sources.yaml")
    release = gpd.read_file(f"zip://{zip_path}!RELEASE.DBF", engine="pyogrio")
    chemical = gpd.read_file(f"zip://{zip_path}!CHEMICAL.DBF", engine="pyogrio")
    seed = _standardize(pd.read_csv(SEED_RELEASES, dtype=str))
    out = pfas_rtns(pd.DataFrame(release), pd.DataFrame(chemical), seed)
    if coordinates:
        # Reuse lookups from the previous build (including RTNs MassDEP has no coordinates for);
        # only query RTNs that are new to the list.
        known = pd.DataFrame(columns=["rtn", "lat", "lon"])
        if RELEASE_LIST.exists() and not refresh_coordinates:
            known = pd.read_csv(RELEASE_LIST, dtype={"rtn": str})[["rtn", "lat", "lon"]]
        todo = [r for r in out["rtn"] if r not in set(known["rtn"])]
        coords = pd.concat([known, fetch_coordinates(todo)], ignore_index=True)
        out = out.merge(coords.drop_duplicates("rtn"), on="rtn", how="left")
    RELEASE_LIST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RELEASE_LIST, index=False)
    record_source(Path(zip_path), len(out))
    log.info("wrote %d PFAS releases to %s", len(out), RELEASE_LIST)
    return RELEASE_LIST


def load_releases() -> pd.DataFrame:
    """One row per PFAS RTN: the built release list if present, else the 2021 seed list."""
    path = RELEASE_LIST if RELEASE_LIST.exists() else SEED_RELEASES
    if path == SEED_RELEASES:
        log.warning("no release list at %s; using 2021 seed list (%s)", RELEASE_LIST, SEED_RELEASES.name)
    df = _standardize(pd.read_csv(path, dtype={"rtn": str}))
    df = df[df["chemical"].map(is_pfas)]
    agg = {c: "first" for c in df.columns if c not in ("rtn", "chemical")}
    agg["chemical"] = _join_chemicals
    out = df.groupby("rtn", as_index=False).agg(agg)
    out["notification_date"] = pd.to_datetime(out.get("notification_date"), errors="coerce")
    out.attrs["source"] = path.name
    return out


TOWN_SLACK_M = 1_000


def outside_town(rel: pd.DataFrame, pts: gpd.GeoSeries, towns: gpd.GeoDataFrame) -> pd.Series:
    """True where a point lies more than TOWN_SLACK_M outside the release's stated town.

    Catches bad published coordinates (e.g. a Barnstable site placed 65 km away). Releases whose
    town field is a village or otherwise unknown can't be checked and count as inside.
    """
    shapes = towns.assign(key=towns["TOWN"].map(normalize_town)).dissolve("key").geometry
    key = rel["town"].map(normalize_town)
    bad = [k in shapes.index and shapes[k].distance(p) > TOWN_SLACK_M for k, p in zip(key, pts)]
    return pd.Series(bad, index=rel.index)


def locate_releases(releases: pd.DataFrame | None = None, index: pd.DataFrame | None = None,
                    towns: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Place each release: MassDEP coordinates where published and plausible, else address matching."""
    rel = load_releases() if releases is None else releases
    has = rel[["lat", "lon"]].notna().all(axis=1) if {"lat", "lon"} <= set(rel.columns) else pd.Series(False, rel.index)
    if has.any():
        pts = gpd.GeoSeries(gpd.points_from_xy(rel.loc[has, "lon"], rel.loc[has, "lat"]), crs=WGS84,
                            index=rel.index[has]).to_crs(CRS)
        bad = outside_town(rel[has], pts, read("towns") if towns is None else towns)
        if bad.any():
            log.warning("MassDEP coordinates outside the stated town, using address instead: %s",
                        ", ".join(rel.loc[bad[bad].index, "rtn"]))
        has.loc[bad[bad].index] = False
    placed = rel[has].copy()
    if len(placed):
        pts = gpd.GeoSeries(gpd.points_from_xy(placed["lon"], placed["lat"]), crs=WGS84).to_crs(CRS)
        placed["x"], placed["y"], placed["geocode_precision"] = pts.x.values, pts.y.values, "massdep"
    rest = geocode(rel[~has], index=index) if (~has).any() else rel.iloc[:0]
    out = pd.concat([placed, rest]).loc[rel.index]
    log.info("located %d releases: %s", len(out), out["geocode_precision"].value_counts().to_dict())
    out = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["x"], out["y"]), crs=CRS)
    out.attrs["source"] = rel.attrs.get("source")
    return out
