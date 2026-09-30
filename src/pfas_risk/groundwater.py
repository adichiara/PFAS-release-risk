"""Groundwater PFAS model: where private wells are likely to draw PFAS-contaminated water.

About a million Massachusetts residents drink from private wells, which no public program
tests. Public groundwater wells were tested systematically (every community and
non-transient system under the 2020 PFAS6 standard), so their raw-water results show how PFAS
in groundwater relates to what is around a well. This module learns that relationship and
applies it where people use private wells.

Training data: the highest raw (untreated) PFAS6 result at each public groundwater source.
Samples are tied to a source by their sampling-point code (``RW-03G`` -> ``<system>-03G``), or,
for systems with a single groundwater source, to that source.

What predicts a well's PFAS (town-grouped cross-validation over about 1,030 wells): measured
PFAS at other systems' wells within a few kilometers (AUC 0.71 for PFAS6 >= 20 ng/L within
3 km) and, more weakly, development around the well (population within 1 km, AUC 0.63).
Proximity to mapped hazard sources (industries, airports, landfills, fire stations and so on),
counted around the well or inside its Zone II recharge area, added nothing beyond
development, so the model uses:

- ``neigh_log_pfas6``: inverse-distance-weighted log(1 + PFAS6) at public wells within 5 km,
  excluding the well's own system, and ``n_neigh`` (0 where there are none);
- ``pop_1km``, ``sewered`` (inside a sewer service area; outside it homes use septic systems)
  and ``in_aquifer`` (high- or medium-yield aquifer).

Private wells take the same features at their census block, with every public well as a
potential neighbour. Public wells are sparse in rural areas, so many private-well blocks have
few neighbours and lean on the weaker development features.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import BallTree

from .config import CRS
from .features import point_sources
from .model import _scaled, town_folds
from .pfas_sources import airport_points, military_areas, tri_points
from .sources import download, read

log = logging.getLogger(__name__)

RADII_M = (1_000, 3_000)
THRESHOLDS = {"over20": 20.0, "detect": 2.0}   # PFAS6 ng/L: state standard; reporting limit


# ---- Training data --------------------------------------------------------------------

def well_results() -> gpd.GeoDataFrame:
    """Public groundwater sources with their highest raw-water PFAS6 result (ng/L)."""
    d = pd.read_csv(download("massdep_dw_pfas"), dtype=str,
                    usecols=["PWSId", "RaworFinished", "SampleLocCode", "Result"])
    r = d[d["RaworFinished"] == "R"].copy()
    r["value"] = pd.to_numeric(r["Result"], errors="coerce").fillna(0.0)
    r["pws"] = r["PWSId"].str.zfill(7)
    code = r["SampleLocCode"].str.strip().str.upper()
    parts = code.str.extract(r"(?:^|[^0-9])0*(\d{1,2})([GS])(?:$|[^A-Z0-9])").fillna(
        code.str.extract(r"^0*(\d{1,2})([GS])"))
    r["source_id"] = r["pws"] + "-" + parts[0].str.zfill(2) + parts[1]

    src = read("pws_sources").drop_duplicates("SOURCE_ID")
    gw = src[src["TYPE"].isin(["GW", "NTNC", "TNC"])]
    matched = r["source_id"].isin(gw["SOURCE_ID"])
    single = gw[gw.groupby("PWS_ID")["SOURCE_ID"].transform("size") == 1].set_index("PWS_ID")["SOURCE_ID"]
    rest = r[~matched & r["pws"].isin(single.index)].assign(source_id=lambda x: x["pws"].map(single))
    best = pd.concat([r[matched], rest]).groupby("source_id")["value"].max()
    wells = gw.set_index("SOURCE_ID").loc[best.index, ["PWS_ID", "TOWN", "TYPE", "geometry"]]
    return gpd.GeoDataFrame(wells.assign(pfas6_max=best), geometry="geometry", crs=CRS)


# ---- Point features -------------------------------------------------------------------

def hazard_points() -> dict[str, gpd.GeoDataFrame]:
    """Point hazard layers: the release model's sources plus airports and TRI reporters,
    which are plausible PFAS sources whether or not they predicted reported releases."""
    layers = {k: v for k, v in point_sources().items() if not k.startswith("pws_")}
    return layers | {"airport": airport_points(), "tri": tri_points()}


def _xy(g: gpd.GeoSeries) -> np.ndarray:
    return np.column_stack([g.x, g.y])


def point_features(points: gpd.GeoSeries) -> pd.DataFrame:
    """Hazard counts, nearest distances and setting at each point (EPSG:26986)."""
    xy = _xy(points)
    out = {}
    for name, layer in hazard_points().items():
        g = layer.geometry
        g = g[g.notna() & ~g.is_empty]
        tree = BallTree(_xy(g.centroid))
        for r in RADII_M:
            out[f"n{r // 1000}k_{name}"] = tree.query_radius(xy, r=r, count_only=True)
        out[f"d_{name}_km"] = tree.query(xy, k=1)[0][:, 0] / 1000

    pts = gpd.GeoDataFrame(geometry=points.values, crs=CRS)
    for name, polys in [("military", military_areas()), ("landfill", read("landfills"))]:
        d = gpd.sjoin_nearest(pts, polys[["geometry"]], distance_col="d")["d"]
        out[f"d_{name}_km"] = d[~d.index.duplicated()].sort_index().to_numpy() / 1000
    aq = read("aquifers")
    aq = aq[aq["TYPE"].isin(["HIGH", "MED"])][["geometry"]]
    out["in_aquifer"] = pts.index.isin(gpd.sjoin(pts, aq, predicate="within").index).astype(float)
    sewer = read("sewer_service")[["geometry"]]
    out["sewered"] = pts.index.isin(gpd.sjoin(pts, sewer, predicate="within").index).astype(float)

    blocks = read("census_blocks", columns=["POP20"])
    btree = BallTree(_xy(blocks.representative_point()))
    idx = btree.query_radius(xy, r=1_000)
    pop = blocks["POP20"].to_numpy()
    out["pop_1km"] = np.array([pop[i].sum() for i in idx], dtype=float)
    return pd.DataFrame(out, index=points.index)


def neighbour_evidence(points: gpd.GeoSeries, wells: gpd.GeoDataFrame, own_pws: np.ndarray | None = None,
                       radius_m: float = 5_000) -> pd.DataFrame:
    """Inverse-distance-weighted log(1 + PFAS6) at public wells within ``radius_m``.

    ``own_pws`` (one system ID per point) excludes a point's own system, so a public well is
    never described by its own or its sister wells' results.
    """
    tree = BallTree(_xy(wells.geometry))
    values = np.log1p(wells["pfas6_max"].to_numpy())
    pws = wells["PWS_ID"].to_numpy()
    idx, dist = tree.query_radius(_xy(points), r=radius_m, return_distance=True)
    level, count = np.zeros(len(points)), np.zeros(len(points))
    for i, (ix, ds) in enumerate(zip(idx, dist, strict=True)):
        keep = pws[ix] != own_pws[i] if own_pws is not None else np.ones(len(ix), bool)
        if keep.any():
            w = 1 / np.maximum(ds[keep], 200.0) ** 2
            level[i] = (values[ix][keep] * w).sum() / w.sum()
            count[i] = keep.sum()
    return pd.DataFrame({"neigh_log_pfas6": level, "n_neigh": count}, index=points.index)


def setting_features(points: gpd.GeoSeries) -> pd.DataFrame:
    """Population within 1 km, sewer service and aquifer at each point."""
    pts = gpd.GeoDataFrame(geometry=points.values, crs=CRS)
    aq = read("aquifers")
    aq = aq[aq["TYPE"].isin(["HIGH", "MED"])][["geometry"]]
    sewer = read("sewer_service")[["geometry"]]
    blocks = read("census_blocks", columns=["POP20"])
    idx = BallTree(_xy(blocks.representative_point())).query_radius(_xy(points), r=1_000)
    pop = blocks["POP20"].to_numpy()
    return pd.DataFrame({
        "pop_1km": [float(pop[i].sum()) for i in idx],
        "sewered": pts.index.isin(gpd.sjoin(pts, sewer, predicate="within").index).astype(float),
        "in_aquifer": pts.index.isin(gpd.sjoin(pts, aq, predicate="within").index).astype(float),
    }, index=points.index)


FEATURES = ["neigh_log_pfas6", "n_neigh", "pop_1km", "sewered", "in_aquifer"]


def training_table() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    wells = well_results()
    X = setting_features(wells.geometry).join(
        neighbour_evidence(wells.geometry, wells, own_pws=wells["PWS_ID"].to_numpy()))
    return X[FEATURES], wells


# ---- Model ----------------------------------------------------------------------------

def models() -> dict[str, object]:
    return {
        # Unweighted, so predicted probabilities stay calibrated to the observed rates.
        "logistic": _scaled(LogisticRegression(C=0.1, max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                            l2_regularization=1.0, class_weight="balanced"),
    }


def evaluate(X: pd.DataFrame, y: np.ndarray, towns: pd.Series, repeats: int = 5) -> pd.DataFrame:
    """Town-grouped 5-fold CV. Baseline: population within 1 km alone."""
    rows = []
    specs = models() | {"baseline_population": _scaled(LogisticRegression(max_iter=1000))}
    for rep in range(repeats):
        folds = town_folds(towns.reset_index(drop=True), 5, seed=rep)
        for name, est in specs.items():
            cols = ["pop_1km"] if name.startswith("baseline") else list(X.columns)
            p = np.zeros(len(y))
            for tr, te in folds:
                p[te] = clone(est).fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[te][cols])[:, 1]
            rows.append({"model": name, "repeat": rep, "roc_auc": roc_auc_score(y, p),
                         "avg_precision": average_precision_score(y, p), "prevalence": y.mean()})
    return pd.DataFrame(rows)


def fit(X: pd.DataFrame, y: np.ndarray, name: str = "logistic") -> object:
    return clone(models()[name]).fit(X, y)


def private_well_estimates(blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    """P(PFAS6 >= 20 ng/L) and P(detected) for groundwater at each block's location."""
    X, wells = training_table()
    pts = blocks.representative_point()
    Xb = setting_features(pts).join(neighbour_evidence(pts, wells))[FEATURES]
    out = {}
    for name, threshold in THRESHOLDS.items():
        m = fit(X, (wells["pfas6_max"].to_numpy() >= threshold).astype(int))
        out[f"p_{name}"] = m.predict_proba(Xb)[:, 1]
    return pd.DataFrame(out, index=blocks.index).assign(n_neigh=Xb["n_neigh"].to_numpy())
