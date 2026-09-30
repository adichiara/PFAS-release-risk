"""Self-contained interactive Leaflet map of unit risk and known releases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .config import WGS84
from .model import risk_density

TEMPLATE = Path(__file__).with_name("map_template.html")


def _round_geojson(gdf: gpd.GeoDataFrame, decimals: int = 5) -> dict:
    gdf = gdf.copy()
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 10 ** -decimals)
    return json.loads(gdf.to_json(drop_id=True))


def compact_units(units: gpd.GeoDataFrame, hex_km2: float | None = None, decimals: int = 5) -> dict:
    """Units as compact rows, so a 1 km² grid (about 22,000 cells) stays a reasonable page size.

    Each row is [id, town index, population, land km², releases, percentile, shape]. ``shape``
    is the [lon, lat] center of a full hexagon, which the page draws itself (``hex_side_m``),
    or a GeoJSON geometry for block groups and hexagons clipped at the coast or state line.
    """
    towns = sorted(units["town"].astype(str).unique())
    town_index = {t: i for i, t in enumerate(towns)}
    side = math.sqrt(2 * hex_km2 * 1e6 / (3 * math.sqrt(3))) if hex_km2 else None
    full = (units.area >= 0.999 * hex_km2 * 1e6).to_numpy() if hex_km2 else np.zeros(len(units), bool)
    centers = gpd.GeoSeries(units.centroid, crs=units.crs).to_crs(WGS84)
    shapes = shapely.set_precision(units.geometry.simplify(25).to_crs(WGS84).values, 10 ** -decimals)
    rows = []
    for i, (uid, r) in enumerate(units.iterrows()):
        shape = ([round(centers.iloc[i].x, decimals), round(centers.iloc[i].y, decimals)] if full[i]
                 else shapely.geometry.mapping(shapes[i]))
        rows.append([str(uid), town_index[str(r["town"])], int(r["POP20"]), round(float(r["land_km2"]), 2),
                     int(r["releases"]), float(r["pct"]), shape])
    return {"towns": towns, "hex_side_m": side, "rows": rows}


BASELINE_TEXT = {
    "bg": "The baseline uses block-group land area only. Larger areas see more reported releases, "
          "so the useful signal is how far the model gets beyond it.",
    "hex": "Cells have equal area, so the baseline uses population and housing density. Denser "
           "places see more reported releases, so the useful signal is how far the model gets beyond it.",
}


def risk_map(features: gpd.GeoDataFrame, scores: pd.Series, releases: gpd.GeoDataFrame,
             model: str, metrics: dict[str, float], baseline: dict[str, float], out_path: Path,
             source_note: str, unit_labels: tuple[str, str] = ("block group", "block groups"),
             baseline_text: str = BASELINE_TEXT["bg"], hex_km2: float | None = None) -> Path:
    """Write a single HTML file. ``scores`` are out-of-fold P(release) per unit."""
    bg = features[["geometry", "town", "POP20", "land_km2", "releases"]].copy()
    bg["score"] = scores.reindex(bg.index)
    # Percentile of risk per km², so small units are comparable to large ones.
    bg["pct"] = risk_density(bg["score"], bg).rank(pct=True).mul(100).round(1)
    cells = compact_units(bg, hex_km2)

    rel = releases[releases["x"].notna()].to_crs(WGS84)
    cols = [c for c in ["rtn", "site_name", "town", "address", "notification_date", "chemical",
                        "geocode_precision"] if c in rel]
    rel = rel[[*cols, "geometry"]].copy()
    if "notification_date" in rel:
        rel["notification_date"] = pd.to_datetime(rel["notification_date"]).dt.strftime("%Y-%m-%d")
    points = _round_geojson(rel.fillna(""))

    page = (TEMPLATE.read_text()
            .replace("__CELLS__", json.dumps(cells, separators=(",", ":")))
            .replace("__POINTS__", json.dumps(points, separators=(",", ":")))
            .replace("__MODEL__", model)
            .replace("__METRICS__", json.dumps({k: [round(float(v), 3), round(float(baseline[k]), 3)]
                                               for k, v in metrics.items()}))
            .replace("__SOURCE__", source_note)
            .replace("__BASELINE_TEXT__", baseline_text)
            .replace("__UNITS__", unit_labels[1])
            .replace("__UNIT__", unit_labels[0])
            .replace("__N_RELEASES__", str(int(np.sum(bg["releases"])))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path
