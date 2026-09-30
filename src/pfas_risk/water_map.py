"""Interactive drinking-water map: public systems by measured PFAS6, private-well areas by estimate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .config import CRS, WGS84
from .drinking_water import service_areas, system_levels
from .features import build_features

TEMPLATE = Path(__file__).with_name("water_map_template.html")
CELL_KM2 = 1.0
# Estimated chance of PFAS6 >= 20 ng/L at private wells (after calibration to MassDEP testing).
PRIVATE_BANDS = ["Under 5%", "5 to 10%", "10 to 20%", "20% or more"]


def _shape(geom, decimals: int = 4) -> dict:
    # Pointwise rounding: display only, and it never fails on slivers left by clipping.
    return shapely.geometry.mapping(shapely.set_precision(geom, 10 ** -decimals, mode="pointwise"))


def _num(x: float, digits: int = 1) -> float | None:
    return None if pd.isna(x) else round(float(x), digits)


def system_rows() -> list[list]:
    sa = service_areas()
    sa = sa.join(system_levels(sa["PWS_ID"]), on="PWS_ID")
    sa = sa.sort_values("PWSPOP_WIN")      # small systems drawn last, on top of large ones
    geoms = sa.geometry.simplify(30).to_crs(WGS84)
    return [[str(r.PWS_NAME).title(), r.PWS_ID, int(r.PWSPOP_WIN), _num(r.current), _num(r.peak),
             r.source if isinstance(r.source, str) else None, r.via if isinstance(r.via, str) else None,
             _shape(g)] for r, g in zip(sa.itertuples(), geoms, strict=True)]


def private_rows(blocks: gpd.GeoDataFrame) -> dict:
    """1 km² cells holding private-well residents, with population-weighted estimates."""
    prv = blocks[blocks["private_well"]]
    cells = build_features(unit="hex", cell_km2=CELL_KM2)[["geometry"]]
    pts = gpd.GeoDataFrame(prv[["POP20", "p_over20", "p_detect", "n_neigh"]],
                           geometry=prv.representative_point(), crs=CRS)
    j = gpd.sjoin(pts, cells, predicate="within")
    j["w20"], j["wdet"] = j["POP20"] * j["p_over20"], j["POP20"] * j["p_detect"]
    g = j.groupby("geoid").agg(pop=("POP20", "sum"), w20=("w20", "sum"), wdet=("wdet", "sum"),
                               neigh=("n_neigh", "median"))
    g = g[g["pop"] > 0]
    side = math.sqrt(2 * CELL_KM2 * 1e6 / (3 * math.sqrt(3)))
    c = cells.loc[g.index].copy()
    # Show private-well cells only outside public water service, so the two layers never overlap.
    served = shapely.union_all(service_areas().geometry.values)
    touches = c.intersects(served).to_numpy()
    c.loc[touches, "geometry"] = c.geometry[touches].difference(served).make_valid()
    keep = ~c.geometry.is_empty
    c, g = c[keep], g[keep.to_numpy()]
    touches = touches[keep.to_numpy()]
    full = (~touches) & (c.area >= 0.999 * CELL_KM2 * 1e6).to_numpy()
    centers = gpd.GeoSeries(c.centroid, crs=CRS).to_crs(WGS84)
    shapes = c.geometry.simplify(25).to_crs(WGS84)
    rows = []
    for i, (cid, r) in enumerate(g.iterrows()):
        shape = ([round(centers.iloc[i].x, 5), round(centers.iloc[i].y, 5)] if full[i] else _shape(shapes.loc[cid], 5))
        rows.append([round(float(r["pop"])), round(r["w20"] / r["pop"], 3), round(r["wdet"] / r["pop"], 3),
                     int(r["neigh"]), shape])
    return {"hex_side_m": side, "rows": rows}


def release_points(releases: gpd.GeoDataFrame) -> dict:
    rel = releases[releases["x"].notna()].to_crs(WGS84)
    rel = rel[[c for c in ["rtn", "site_name", "town"] if c in rel] + ["geometry"]].fillna("")
    rel["geometry"] = shapely.set_precision(rel.geometry.values, 1e-5)
    return json.loads(rel.to_json(drop_id=True))


def water_map(blocks: gpd.GeoDataFrame, releases: gpd.GeoDataFrame, out_path: Path) -> Path:
    page = (TEMPLATE.read_text()
            .replace("__SYSTEMS__", json.dumps(system_rows(), separators=(",", ":")))
            .replace("__PRIVATE__", json.dumps(private_rows(blocks), separators=(",", ":")))
            .replace("__RELEASES__", json.dumps(release_points(releases), separators=(",", ":"))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path


def private_band_counts(blocks: pd.DataFrame) -> dict[str, int]:
    """Private-well residents by estimated chance of PFAS6 >= 20 ng/L (the map's bands)."""
    prv = blocks[blocks["private_well"]]
    bands = pd.cut(prv["p_over20"], [-np.inf, 0.05, 0.1, 0.2, np.inf],
                   labels=PRIVATE_BANDS, right=False)
    return {str(k): round(float(v)) for k, v in prv["POP20"].groupby(bands, observed=False).sum().items()}
