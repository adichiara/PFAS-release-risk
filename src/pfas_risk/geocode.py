"""Offline geocoding of Massachusetts street addresses against MassGIS address points.

Matching is deterministic and reproducible (no web service):

1. ``address``  - house number, street and town all match an address point.
2. ``street``   - street and town match; uses the point on that street whose house
                  number is closest (or the street's median point if no number).
3. unmatched    - left without coordinates; add a row to
                  data/seed/geocode_overrides.csv to place it by hand.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pyogrio

from .config import INTERIM_DIR, SEED_DIR
from .sources import download, source

log = logging.getLogger(__name__)

INDEX_PATH = INTERIM_DIR / "address_index.parquet"
OVERRIDES_PATH = SEED_DIR / "geocode_overrides.csv"

_SUFFIXES = {
    "STREET": "ST", "STR": "ST", "ROAD": "RD", "AVENUE": "AVE", "AV": "AVE", "DRIVE": "DR",
    "LANE": "LN", "COURT": "CT", "CIRCLE": "CIR", "PLACE": "PL", "TERRACE": "TER",
    "BOULEVARD": "BLVD", "PARKWAY": "PKWY", "HIGHWAY": "HWY", "TURNPIKE": "TPKE",
    "SQUARE": "SQ", "EXTENSION": "EXT", "ROUTE": "RTE", "RT": "RTE", "MOUNT": "MT",
    "SAINT": "ST", "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
}


def normalize_street(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^A-Z0-9 ]", " ", s.upper())
    return " ".join(_SUFFIXES.get(t, t) for t in s.split())


def normalize_town(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^A-Z ]", " ", s.upper()).split())


_QUALIFIER = re.compile(r"^\s*(NEAR|OFF|REAR( OF)?|BEHIND|ACROSS FROM|ADJACENT TO|ADJ(\.)?( TO)?)\s+", re.IGNORECASE)


def split_address(addr: str | None) -> tuple[int | None, str]:
    """'121-125 Liberty Street' -> (121, 'LIBERTY ST'). Number is None if absent.

    Location qualifiers MassDEP uses ("Near 875 Spring St", "Off Fish Rd") and anything after
    a comma ("104 Powdermill Rd, Rear of property") are dropped.
    """
    if not isinstance(addr, str):
        return None, ""
    addr = _QUALIFIER.sub("", addr.split(",")[0])
    m = re.match(r"\s*(\d+)[A-Za-z]?(?:\s*-\s*\d+[A-Za-z]?)?\s+(.*)", addr)
    if m:
        return int(m.group(1)), normalize_street(m.group(2))
    return None, normalize_street(addr)


def build_index(force: bool = False) -> pd.DataFrame:
    """Reduce the 3.7M statewide address points to a small parquet lookup table."""
    if INDEX_PATH.exists() and not force:
        return pd.read_parquet(INDEX_PATH)
    meta = source("address_points")
    path = download("address_points")
    log.info("building address index from %s", path.name)
    df = pyogrio.read_dataframe(
        f"zip://{path}!{meta['gdb']}", layer=meta["layer"],
        columns=["ADDRESS_NUMBER", "STREET_NAME", "GEOGRAPHIC_TOWN", "COMMUNITY_NAME"],
    )
    out = pd.DataFrame({
        "num": pd.to_numeric(df["ADDRESS_NUMBER"], errors="coerce").astype("Int64"),
        "street": df["STREET_NAME"].map(normalize_street),
        "town": df["GEOGRAPHIC_TOWN"].map(normalize_town),
        "community": df["COMMUNITY_NAME"].map(normalize_town),
        "x": df.geometry.x.astype("float32"),
        "y": df.geometry.y.astype("float32"),
    })
    out = out[out.street != ""].drop_duplicates(["num", "street", "town", "community"])
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INDEX_PATH, index=False)
    return out


def geocode(df: pd.DataFrame, address_col: str = "address", town_col: str = "town",
            index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add x, y (EPSG:26986) and geocode_precision columns to ``df``."""
    idx = build_index() if index is None else index
    by_street = {k: g for k, g in idx.groupby(["street", "town"], sort=False)}
    by_street_comm = {k: g for k, g in idx.groupby(["street", "community"], sort=False)}

    def match(addr, town) -> tuple[float, float, str]:
        num, street = split_address(addr)
        t = normalize_town(town)
        cand = by_street.get((street, t))
        if cand is None:  # town field may hold a village name (e.g. "Hyannis")
            cand = by_street_comm.get((street, t))
        if cand is None or not street:
            return np.nan, np.nan, "unmatched"
        exact = cand[cand.num == num] if num is not None else cand.iloc[:0]
        if len(exact):
            return float(exact.x.median()), float(exact.y.median()), "address"
        if num is not None and cand.num.notna().any():
            nearest = cand.loc[(cand.num.astype("float") - num).abs().idxmin()]
            return float(nearest.x), float(nearest.y), "street"
        return float(cand.x.median()), float(cand.y.median()), "street"

    matched = [match(a, t) for a, t in zip(df[address_col], df[town_col], strict=True)]
    out = df.copy()
    out["x"], out["y"], out["geocode_precision"] = (list(col) for col in zip(*matched, strict=True)) if matched else ([], [], [])
    return apply_overrides(out)


def apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-placed coordinates (x/y in EPSG:26986) keyed by RTN win over matches."""
    if not OVERRIDES_PATH.exists() or "rtn" not in df:
        return df
    ov = pd.read_csv(OVERRIDES_PATH, dtype={"rtn": str}).set_index("rtn")
    hit = df["rtn"].isin(ov.index)
    df.loc[hit, "x"] = df.loc[hit, "rtn"].map(ov["x"])
    df.loc[hit, "y"] = df.loc[hit, "rtn"].map(ov["y"])
    df.loc[hit, "geocode_precision"] = "override"
    return df
