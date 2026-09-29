"""The response: MassDEP reportable releases (RTNs) involving PFAS."""

from __future__ import annotations

import logging
import re

import geopandas as gpd
import pandas as pd

from .config import CRS, SEED_RELEASES
from .geocode import geocode
from .sources import local_path

log = logging.getLogger(__name__)

# MassDEP chemical names are free text: "PFAS6 (SUMMATION)", "PERFLUROHEXANSULFONIC ACID PFHXS",
# "TOTAL REGULATED PFAS COMPOUNDS (PPT)", ... Match the family, including common misspellings.
PFAS_PATTERN = re.compile(
    r"PFAS|PFOS|PFOA|PFHX|PFNA|PFDA|PFBS|PFHPA|GENX|HFPO|PERFLU|POLYFLU|FLUOROALKYL|FLUOROTELOMER",
    re.IGNORECASE,
)

# Column names in a MassDEP portal export vary by export type; map what we find.
_COLUMN_ALIASES = {
    "rtn": ["rtn", "release tracking number"],
    "town": ["town", "city", "municipality", "release town"],
    "address": ["address", "release address", "location address", "street address"],
    "site_name": ["site_name", "site name", "location name", "name"],
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


def load_releases() -> pd.DataFrame:
    """One row per PFAS RTN, from a MassDEP export if present, else the 2021 seed list."""
    export = local_path("massdep_releases")
    path = export if export.exists() else SEED_RELEASES
    if path == SEED_RELEASES:
        log.warning("no MassDEP export at %s; using 2021 seed list (%s)", export, SEED_RELEASES.name)
    df = _standardize(pd.read_csv(path, dtype=str))
    df = df[df["chemical"].map(is_pfas)]
    # An RTN can list several chemicals; keep one row, joining the chemical names.
    agg = {c: "first" for c in df.columns if c not in ("rtn", "chemical")}
    agg["chemical"] = lambda s: "; ".join(sorted(set(s.dropna())))
    out = df.groupby("rtn", as_index=False).agg(agg)
    if "notification_date" in out:
        out["notification_date"] = pd.to_datetime(out["notification_date"], errors="coerce")
    out.attrs["source"] = path.name
    return out


def locate_releases(releases: pd.DataFrame | None = None, index: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    rel = load_releases() if releases is None else releases
    rel = geocode(rel, index=index)
    counts = rel["geocode_precision"].value_counts().to_dict()
    log.info("geocoded %d releases: %s", len(rel), counts)
    return gpd.GeoDataFrame(rel, geometry=gpd.points_from_xy(rel["x"], rel["y"]), crs=CRS)
