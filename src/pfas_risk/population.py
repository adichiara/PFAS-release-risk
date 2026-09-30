"""Who lives in higher-risk areas: model scores carried down to 2020 census blocks.

A cell's score says how likely the cell is to contain a reported PFAS release, relative to
other cells; it is not a statement about any one home. Carried to the census blocks inside
the cell, it describes the risk of the area people live in. Blocks are small (most sit inside
one 1 km² cell), so each takes the area-weighted mean of the cells it overlaps.

Blocks are then linked to the 2020 Environmental Justice block groups (minority, income and
limited-English criteria) and to community water service areas: homes outside them are
presumed to be on private wells, which matters for PFAS as much as demographics do.

Two comparisons per group: the share of its residents living in the top-risk 10% (or 25%) of
land, and that share relative to what population density alone would give. Density is a
model input and many demographics follow it, so the second number asks whether a group is in
higher-risk areas beyond living in denser places.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS
from .model import risk_density
from .sources import read

TIERS = {"top10": 0.10, "top25": 0.25}
DENSITY_BANDS = 10


def cell_tiers(df: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Percentile of risk per km² and top-10%/25%-of-land flags per cell (as on the map)."""
    density = risk_density(scores.reindex(df.index), df)
    order = density.sort_values(ascending=False).index
    land = df.loc[order, "land_km2"]
    start = (land.cumsum() - land) / land.sum()      # share of land ranked above this cell
    out = pd.DataFrame({"pct": density.rank(pct=True) * 100}, index=df.index)
    for name, frac in TIERS.items():
        out[name] = (start < frac).reindex(df.index).astype(float)
    return out


def block_risk(cells: gpd.GeoDataFrame, tiers: pd.DataFrame) -> gpd.GeoDataFrame:
    """Census blocks with population, the area-weighted cell percentile and tier shares."""
    blocks = read("census_blocks", columns=["GEOID20", "POP20", "HOUSING20", "ALAND20"])
    blocks = blocks[blocks["POP20"] > 0].copy()
    blocks["block_area"] = blocks.area
    grid = gpd.GeoDataFrame(tiers.join(cells[["geometry"]]), geometry="geometry", crs=CRS)
    pieces = gpd.overlay(blocks[["GEOID20", "block_area", "geometry"]], grid.reset_index(),
                         how="intersection", keep_geom_type=True)
    w = pieces.area / pieces["block_area"]
    cols = ["pct", *TIERS]
    weighted = pieces[cols].mul(w, axis=0).groupby(pieces["GEOID20"]).sum()
    covered = w.groupby(pieces["GEOID20"]).sum()
    blocks = blocks.set_index("GEOID20")
    blocks[cols] = weighted.div(covered, axis=0).reindex(blocks.index)
    return blocks.dropna(subset=["pct"])


def attach_context(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add each block's EJ block group fields and whether it has community public water.

    Blocks outside the EJ block groups get no minority or income figures (the layer covers
    only EJ block groups), so minority residents are estimated inside EJ areas only.
    """
    rep = gpd.GeoDataFrame(geometry=blocks.representative_point(), crs=CRS)
    ej = read("ej_populations")[["GEOID", "MUNICIPALI", "EJ", "EJ_CRIT_DE", "PCT_MINORI",
                                 "LIMENGHHPC", "BG_MHHI", "geometry"]]
    # The layer holds only the EJ block groups, so a block outside every polygon is not EJ.
    j = gpd.sjoin(rep, ej, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = blocks.join(j.drop(columns=["geometry", "index_right"]))
    water = read("water_service_areas")[["geometry"]]
    served = gpd.sjoin(rep, water, predicate="within").index.unique()
    out["public_water"] = out.index.isin(served)
    return out


def _groups(b: pd.DataFrame) -> dict[str, pd.Series]:
    """Estimated residents per block for each group (block population x block-group share)."""
    crit = b["EJ_CRIT_DE"].fillna("")
    ej = (b["EJ"] == "Yes").astype(float)
    return {
        "All residents": b["POP20"],
        "In EJ block groups": b["POP20"] * ej,
        "In EJ block groups (minority criterion)": b["POP20"] * ej * crit.str.contains("minority", case=False),
        "In EJ block groups (income criterion)": b["POP20"] * ej * crit.str.contains("income", case=False),
        "In EJ block groups (English isolation)": b["POP20"] * ej * crit.str.contains("english", case=False),
        "Not in EJ block groups": b["POP20"] * (1 - ej),
        "Outside community water service (private wells)": b["POP20"] * ~b["public_water"],
        "On community public water": b["POP20"] * b["public_water"],
    }


def summarize(b: pd.DataFrame) -> list[dict]:
    """Per group: share of residents in each tier, and that share relative to density alone."""
    density = b["POP20"] / (b["ALAND20"].clip(lower=1) / 1e6)
    # Population-weighted density deciles: each band holds about a tenth of residents.
    order = density.sort_values().index
    cum = b.loc[order, "POP20"].cumsum() / b["POP20"].sum()
    band = pd.Series(np.minimum((cum * DENSITY_BANDS).astype(int), DENSITY_BANDS - 1), index=order)
    band = band.reindex(b.index)
    rows = []
    for name, pop in _groups(b).items():
        total = pop.sum()
        if total == 0:
            continue
        row = {"group": name, "residents": round(float(total))}
        for tier in TIERS:
            share = (pop * b[tier]).sum() / total
            band_rate = (b["POP20"] * b[tier]).groupby(band).sum() / b["POP20"].groupby(band).sum()
            expected = (pop * band.map(band_rate)).sum() / total
            row[tier] = round(float(share), 4)
            row[f"{tier}_vs_density"] = round(float(share / expected), 3) if expected else None
        row["mean_pct"] = round(float((pop * b["pct"]).sum() / total), 1)
        rows.append(row)
    return rows


def block_group_table(b: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted risk for each 2020 block group, with its EJ fields."""
    b = b.assign(bg=b.index.str[:12], w_pct=b["pct"] * b["POP20"],
                 **{f"w_{t}": b[t] * b["POP20"] for t in TIERS},
                 w_private=b["POP20"] * ~b["public_water"])
    g = b.groupby("bg")
    out = pd.DataFrame({
        "town": g["MUNICIPALI"].first(), "population": g["POP20"].sum(),
        "risk_percentile_pop_weighted": (g["w_pct"].sum() / g["POP20"].sum()).round(1),
        **{f"share_in_{t}_land": (g[f"w_{t}"].sum() / g["POP20"].sum()).round(3) for t in TIERS},
        "share_private_wells": (g["w_private"].sum() / g["POP20"].sum()).round(3),
        "ej": g["EJ"].first().fillna("No"), "ej_criteria": g["EJ_CRIT_DE"].first(),
        "pct_minority": g["PCT_MINORI"].first().round(1),
        "pct_limited_english_households": g["LIMENGHHPC"].first().round(1),
        "median_household_income": g["BG_MHHI"].first(),
    })
    out.index.name = "block_group"
    return out.sort_values("risk_percentile_pop_weighted", ascending=False)


def population_report(cells: gpd.GeoDataFrame, scores: pd.Series) -> tuple[list[dict], pd.DataFrame]:
    blocks = attach_context(block_risk(cells, cell_tiers(cells, scores)))
    return summarize(blocks), block_group_table(blocks)
