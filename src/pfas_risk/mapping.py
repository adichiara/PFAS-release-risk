"""Self-contained interactive Leaflet map of block-group risk and known releases."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .config import WGS84

TEMPLATE = Path(__file__).with_name("map_template.html")


def _round_geojson(gdf: gpd.GeoDataFrame, decimals: int = 5) -> dict:
    gdf = gdf.copy()
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 10 ** -decimals)
    return json.loads(gdf.to_json(drop_id=True))


def risk_map(features: gpd.GeoDataFrame, scores: pd.Series, releases: gpd.GeoDataFrame,
             model: str, metrics: dict[str, float], baseline: dict[str, float], out_path: Path,
             source_note: str) -> Path:
    """Write a single HTML file. ``scores`` are out-of-fold P(release) per block group."""
    bg = features[["geometry", "town", "POP20", "land_km2", "releases"]].copy()
    bg["score"] = scores.reindex(bg.index)
    # Percentile of risk per km², so small dense block groups are comparable to large ones.
    bg["pct"] = (bg["score"] / bg["land_km2"]).rank(pct=True).mul(100).round(1)
    bg["geometry"] = bg.geometry.simplify(25)
    bg = bg.to_crs(WGS84).reset_index()
    bg["land_km2"] = bg["land_km2"].round(2)
    polys = _round_geojson(bg[["geoid", "town", "POP20", "land_km2", "releases", "pct", "geometry"]])

    rel = releases[releases["x"].notna()].to_crs(WGS84)
    cols = [c for c in ["rtn", "site_name", "town", "address", "notification_date", "chemical",
                        "geocode_precision"] if c in rel]
    rel = rel[[*cols, "geometry"]].copy()
    if "notification_date" in rel:
        rel["notification_date"] = pd.to_datetime(rel["notification_date"]).dt.strftime("%Y-%m-%d")
    points = _round_geojson(rel.fillna(""))

    page = (TEMPLATE.read_text()
            .replace("__POLYGONS__", json.dumps(polys, separators=(",", ":")))
            .replace("__POINTS__", json.dumps(points, separators=(",", ":")))
            .replace("__MODEL__", model)
            .replace("__METRICS__", json.dumps({k: [round(float(v), 3), round(float(baseline[k]), 3)]
                                               for k, v in metrics.items()}))
            .replace("__SOURCE__", source_note)
            .replace("__N_RELEASES__", str(int(np.sum(bg["releases"])))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path
