"""Spatial units: 2020 census block groups, or an equal-area hexagon grid.

Block groups vary in size by three orders of magnitude, and land area alone explains most
of which ones contain a reported release. Equal-area hexagons remove that: every full cell
has the same exposure, so any remaining signal has to come from what is in and around it.

Both units come back with the same columns (``geoid`` index, land_km2, water_frac, POP20,
HOUSING20, pop_density, housing_density, town) so the rest of the pipeline is unit-agnostic.
"""

from __future__ import annotations

import logging
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .config import CRS
from .sources import read

log = logging.getLogger(__name__)

# Cells keep at least this share of a full hexagon's area after clipping to the state.
MIN_CELL_SHARE = 0.10


def block_groups() -> gpd.GeoDataFrame:
    bg = read("block_groups", columns=["GEOID20", "ALAND20", "AWATER20", "POP20", "HOUSING20"])
    bg = bg.rename(columns={"GEOID20": "geoid"}).set_index("geoid")
    bg["land_km2"] = bg["ALAND20"] / 1e6
    bg["water_frac"] = bg["AWATER20"] / (bg["ALAND20"] + bg["AWATER20"])
    bg["town"] = _town_of(bg)
    return _densities(bg.drop(columns=["ALAND20", "AWATER20"]))


def _densities(u: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    u["pop_density"] = u["POP20"] / u["land_km2"]
    u["housing_density"] = u["HOUSING20"] / u["land_km2"]
    return u


def _town_of(units: gpd.GeoDataFrame) -> pd.Series:
    towns = read("towns", columns=["TOWN"])
    pts = gpd.GeoDataFrame(geometry=units.representative_point(), crs=CRS)
    # Nearest rather than within: a few coastal interior points fall in water just outside
    # the town polygons.
    j = gpd.sjoin_nearest(pts, towns, how="left")
    j = j[~j.index.duplicated()]
    return j["TOWN"].str.title()


def hexagons(bounds: tuple[float, float, float, float], cell_km2: float) -> gpd.GeoDataFrame:
    """Flat-top hexagons of ``cell_km2`` covering ``bounds`` (EPSG:26986 meters)."""
    side = math.sqrt(2 * cell_km2 * 1e6 / (3 * math.sqrt(3)))
    dx, dy = 1.5 * side, math.sqrt(3) * side
    minx, miny, maxx, maxy = bounds
    cols = np.arange(minx - side, maxx + dx, dx)
    rows = np.arange(miny - dy, maxy + dy, dy)
    angles = np.deg2rad(np.arange(0, 360, 60))
    ring = np.column_stack([np.cos(angles), np.sin(angles)]) * side
    polys = []
    for i, x in enumerate(cols):
        offset = dy / 2 if i % 2 else 0.0
        for y in rows:
            polys.append(shapely.Polygon(ring + np.array([x, y + offset])))
    return gpd.GeoDataFrame(geometry=polys, crs=CRS)


def hex_grid(cell_km2: float = 4.0) -> gpd.GeoDataFrame:
    """Equal-area hexagons clipped to Massachusetts, with census attributes by area weighting."""
    towns = read("towns", columns=["TOWN"])
    state = shapely.union_all(towns.geometry.values)
    grid = hexagons(state.bounds, cell_km2)
    grid = grid[grid.intersects(state)].copy()
    grid["geometry"] = grid.intersection(state)
    grid = grid[grid.area >= MIN_CELL_SHARE * cell_km2 * 1e6]
    tag = f"hex{cell_km2:g}"
    grid.index = pd.Index([f"{tag}_{i:05d}" for i in range(len(grid))], name="geoid")

    log.info("hex grid: %d cells of %g km²; weighting census blocks", len(grid), cell_km2)
    blocks = read("census_blocks", columns=["ALAND20", "AWATER20", "POP20", "HOUSING20", "TOWN"])
    blocks["block_area"] = blocks.area
    pieces = gpd.overlay(blocks, grid.reset_index(), how="intersection", keep_geom_type=True)
    share = pieces.area / pieces["block_area"]
    for c in ["ALAND20", "AWATER20", "POP20", "HOUSING20"]:
        pieces[c] = pieces[c] * share
    sums = pieces.groupby("geoid")[["ALAND20", "AWATER20", "POP20", "HOUSING20"]].sum()
    # Majority town by land within the cell (used to hold out whole towns in validation).
    town_land = pieces.groupby(["geoid", "TOWN"])["ALAND20"].sum().reset_index()
    town = town_land.sort_values("ALAND20").drop_duplicates("geoid", keep="last").set_index("geoid")["TOWN"]

    out = grid.join(sums)
    out[["ALAND20", "AWATER20", "POP20", "HOUSING20"]] = out[["ALAND20", "AWATER20", "POP20",
                                                               "HOUSING20"]].fillna(0)
    out = out[out["ALAND20"] > 0]
    out["land_km2"] = out["ALAND20"] / 1e6
    out["water_frac"] = out["AWATER20"] / (out["ALAND20"] + out["AWATER20"])
    out["town"] = town.reindex(out.index).str.title()
    missing = out["town"].isna()
    if missing.any():
        out.loc[missing, "town"] = _town_of(out[missing])
    return _densities(out.drop(columns=["ALAND20", "AWATER20"]))
