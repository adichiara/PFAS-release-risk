"""PFAS-related industry locations from EPA's Facility Registry Service (FRS).

The FRS state file links every facility EPA programs know about to its coordinates and to
the NAICS codes each program reports. config/pfas_sectors.yaml groups NAICS prefixes into
sectors; each sector becomes a point layer for the block-group features.
"""

from __future__ import annotations

import zipfile
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import yaml

from .config import CONFIG_DIR, CRS, WGS84
from .sources import download


@lru_cache
def sector_definitions() -> dict[str, dict]:
    with open(CONFIG_DIR / "pfas_sectors.yaml") as f:
        return yaml.safe_load(f)["sectors"]


def read_frs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(facilities with coordinates, NAICS rows) from the FRS state file."""
    path = download("epa_frs")
    with zipfile.ZipFile(path) as z:
        fac = pd.read_csv(z.open("MA_FACILITY_FILE.CSV"), dtype=str, encoding="latin1",
                          usecols=["REGISTRY_ID", "PRIMARY_NAME", "LATITUDE83", "LONGITUDE83"])
        naics = pd.read_csv(z.open("MA_NAICS_FILE.CSV"), dtype=str, encoding="latin1",
                            usecols=["REGISTRY_ID", "NAICS_CODE"])
    fac[["LATITUDE83", "LONGITUDE83"]] = fac[["LATITUDE83", "LONGITUDE83"]].apply(pd.to_numeric, errors="coerce")
    fac = fac.dropna(subset=["LATITUDE83", "LONGITUDE83"]).drop_duplicates("REGISTRY_ID")
    return fac, naics


def sector_facilities(fac: pd.DataFrame, naics: pd.DataFrame,
                      sectors: dict[str, dict] | None = None) -> dict[str, gpd.GeoDataFrame]:
    """One point layer (EPSG:26986) per sector; a facility can belong to several sectors."""
    sectors = sector_definitions() if sectors is None else sectors
    out = {}
    for name, spec in sectors.items():
        prefixes = tuple(str(p) for p in spec["naics"])
        ids = naics.loc[naics["NAICS_CODE"].str.startswith(prefixes, na=False), "REGISTRY_ID"].unique()
        f = fac[fac["REGISTRY_ID"].isin(ids)]
        pts = gpd.GeoDataFrame(f[["REGISTRY_ID", "PRIMARY_NAME"]],
                               geometry=gpd.points_from_xy(f["LONGITUDE83"], f["LATITUDE83"]), crs=WGS84)
        out[f"ind_{name}"] = pts.to_crs(CRS)
    return out


def industry_points() -> dict[str, gpd.GeoDataFrame]:
    return sector_facilities(*read_frs())
