"""Block-group feature table.

Every point source gets three measures so no single boundary choice drives the model:
``n_<src>`` (count inside the block group), ``n2k_<src>`` (count within 2 km of it) and
``d_<src>_km`` (distance from the block group's interior point to the nearest one).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS, INTERIM_DIR
from .industry import industry_points
from .sources import read

log = logging.getLogger(__name__)

FEATURES_PATH = INTERIM_DIR / "features.parquet"
NEIGHBORHOOD_M = 2_000


def block_groups() -> gpd.GeoDataFrame:
    bg = read("block_groups", columns=["GEOID20", "ALAND20", "AWATER20", "POP20", "HOUSING20"])
    bg = bg.rename(columns={"GEOID20": "geoid"}).set_index("geoid")
    bg["land_km2"] = bg["ALAND20"] / 1e6
    bg["water_frac"] = bg["AWATER20"] / (bg["ALAND20"] + bg["AWATER20"])
    bg["pop_density"] = bg["POP20"] / bg["land_km2"]
    bg["housing_density"] = bg["HOUSING20"] / bg["land_km2"]
    bg["town"] = _town_of(bg)
    return bg.drop(columns=["ALAND20", "AWATER20"])


def _town_of(bg: gpd.GeoDataFrame) -> pd.Series:
    towns = read("towns", columns=["TOWN"])
    pts = gpd.GeoDataFrame(geometry=bg.representative_point(), crs=CRS)
    # Nearest rather than within: a few coastal block groups' interior points fall in
    # water just outside the town polygons.
    j = gpd.sjoin_nearest(pts, towns, how="left")
    j = j[~j.index.duplicated()]
    return j["TOWN"].str.title()


def point_sources() -> dict[str, gpd.GeoDataFrame]:
    bwp = read("bwp_major")
    return industry_points() | {
        "fire_station": read("fire_stations"),
        "bwp_major": bwp,
        "haz_waste_lqg": bwp[(bwp["LQG_RCRA"] == "Y") | (bwp["LQG_MA"] == "Y")],
        "air_permit": bwp[bwp["AIR"] == "Y"],
        "ust": read("ust"),
    }


def point_measures(bg: gpd.GeoDataFrame, pts: gpd.GeoDataFrame, name: str) -> pd.DataFrame:
    pts = pts[~pts.geometry.is_empty & pts.geometry.notna()][["geometry"]]
    inside = gpd.sjoin(pts, bg[["geometry"]], predicate="within").groupby("geoid").size()
    ring = gpd.GeoDataFrame(geometry=bg.buffer(NEIGHBORHOOD_M), crs=CRS)
    near = gpd.sjoin(pts, ring, predicate="within").groupby("geoid").size()
    rep = gpd.GeoDataFrame(geometry=bg.representative_point(), crs=CRS)
    nearest = gpd.sjoin_nearest(rep, pts, distance_col="d")["d"]
    nearest = nearest[~nearest.index.duplicated()]
    return pd.DataFrame({
        f"n_{name}": inside.reindex(bg.index, fill_value=0),
        f"n2k_{name}": near.reindex(bg.index, fill_value=0),
        f"d_{name}_km": nearest.reindex(bg.index) / 1000,
    })


def area_fraction(bg: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, name: str) -> pd.Series:
    polys = gpd.GeoDataFrame(geometry=[polys.union_all()], crs=CRS)
    inter = gpd.overlay(bg.reset_index()[["geoid", "geometry"]], polys, how="intersection")
    frac = inter.set_index("geoid").area / bg.area
    return frac.groupby(level=0).sum().reindex(bg.index, fill_value=0).clip(0, 1).rename(name)


def line_density(bg: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, name: str) -> pd.Series:
    # Lines on the left so the intersection keeps line geometries.
    inter = gpd.overlay(lines[["geometry"]], bg.reset_index()[["geoid", "geometry"]],
                        how="intersection", keep_geom_type=True)
    km = (inter.length / 1000).groupby(inter["geoid"]).sum()
    return (km.reindex(bg.index, fill_value=0) / bg["land_km2"]).rename(name)


def build_features(force: bool = False) -> gpd.GeoDataFrame:
    if FEATURES_PATH.exists() and not force:
        return gpd.read_parquet(FEATURES_PATH)
    bg = block_groups()
    parts = [bg]
    for name, pts in point_sources().items():
        log.info("features: %s (%d points)", name, len(pts))
        parts.append(point_measures(bg, pts, name))

    landfills = read("landfills")
    rep = gpd.GeoDataFrame(geometry=bg.representative_point(), crs=CRS)
    d = gpd.sjoin_nearest(rep, landfills[["geometry"]], distance_col="d")["d"]
    parts.append((d[~d.index.duplicated()] / 1000).rename("d_landfill_km").to_frame())
    parts.append(area_fraction(bg, landfills, "landfill_frac").to_frame())

    aq = read("aquifers")
    parts.append(area_fraction(bg, aq[aq["TYPE"].isin(["HIGH", "MED"])], "aquifer_frac").to_frame())
    parts.append(line_density(bg, read("major_roads"), "major_road_km_per_km2").to_frame())

    out = gpd.GeoDataFrame(pd.concat(parts, axis=1), geometry="geometry", crs=CRS)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH)
    return out


def attach_releases(features: gpd.GeoDataFrame, releases: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``releases`` (count of located PFAS RTNs) to the feature table."""
    located = releases[releases["x"].notna()]
    hits = gpd.sjoin(located[["rtn", "geometry"]], features[["geometry"]], predicate="within")
    counts = hits.groupby("geoid").size()
    out = features.copy()
    out["releases"] = counts.reindex(out.index, fill_value=0).astype(int)
    out.attrs["unlocated_releases"] = int(releases["x"].isna().sum())
    out.attrs["releases_outside_units"] = int(len(located) - len(hits))
    return out


def model_columns(df: pd.DataFrame) -> list[str]:
    skip = {"geometry", "town", "releases", "POP20", "HOUSING20"}
    return [c for c in df.columns if c not in skip and np.issubdtype(df[c].dtype, np.number)]
