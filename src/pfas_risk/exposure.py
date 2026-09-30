"""Drinking-water PFAS exposure for every Massachusetts resident, by census block.

Residents on public water get their community system's measured PFAS6 (``drinking_water``).
Residents on private wells, which no public program tests, get the groundwater model's
probability that water at their location has PFAS6 at or above the 20 ng/L state standard,
and that PFAS is detected at all (``groundwater``). Blocks are linked to the 2020
Environmental Justice block groups for the breakdown by group.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from .drinking_water import (
    MCL_NG_L,
    band,
    block_water,
    finished_results,
    service_areas,
    system_levels,
    ucmr5_levels,
)
from .groundwater import THRESHOLDS, evaluate, private_well_estimates, training_table
from .population import attach_context
from .sources import read

log = logging.getLogger(__name__)

MWRA_PWS = "6000000"   # Massachusetts Water Resources Authority (wholesale supplier)


def block_exposure() -> gpd.GeoDataFrame:
    blocks = read("census_blocks", columns=["GEOID20", "POP20", "ALAND20", "TOWN"])
    blocks = blocks[blocks["POP20"] > 0].set_index("GEOID20")
    log.info("exposure: %d populated blocks", len(blocks))
    water = block_water(blocks)
    b = attach_context(blocks).join(water[["pws", "system", "current", "peak", "source", "private_well"]])
    private = b["private_well"]
    log.info("exposure: groundwater estimates for %d private-well blocks", int(private.sum()))
    est = private_well_estimates(b[private])
    return b.join(est)


def _share(pop: pd.Series, mask: pd.Series) -> float:
    total = pop.sum()
    return round(float((pop * mask).sum() / total), 4) if total else float("nan")


def summarize(b: pd.DataFrame) -> list[dict]:
    """Per group: public-water PFAS6 shares and private-well estimates."""
    ej = b["EJ"] == "Yes"
    crit = b["EJ_CRIT_DE"].fillna("")
    groups = {
        "All residents": pd.Series(True, index=b.index),
        "In EJ block groups": ej,
        "EJ: minority criterion": ej & crit.str.contains("minority", case=False),
        "EJ: income criterion": ej & crit.str.contains("income", case=False),
        "EJ: English isolation": ej & crit.str.contains("english", case=False),
        "Not in EJ block groups": ~ej,
    }
    rows = []
    public = ~b["private_well"]
    for name, g in groups.items():
        pop = b["POP20"].where(g, 0.0)
        pub, prv = pop.where(public, 0.0), pop.where(~public, 0.0)
        rows.append({
            "group": name,
            "residents": round(float(pop.sum())),
            "public_water": round(float(pub.sum())),
            "current_over20": _share(pub, b["current"] >= MCL_NG_L),
            "current_10_20": _share(pub, (b["current"] >= 10) & (b["current"] < MCL_NG_L)),
            "current_detected": _share(pub, b["current"] >= 2),
            "peak_over20": _share(pub, b["peak"] >= MCL_NG_L),
            "private_wells": round(float(prv.sum())),
            "private_expected_over20": _share(prv, b["p_over20"].fillna(0)),
            "private_expected_detected": _share(prv, b["p_detect"].fillna(0)),
        })
    return rows


def headline(b: pd.DataFrame) -> dict:
    public = ~b["private_well"]
    pop = b["POP20"]
    pub = pop[public]
    return {
        "residents": round(float(pop.sum())),
        "public_water": round(float(pub.sum())),
        "public_by_current_band": {k: round(float(v)) for k, v in
                                   pub.groupby(b.loc[public, "current"].map(band)).sum().items()},
        "public_by_peak_band": {k: round(float(v)) for k, v in
                                pub.groupby(b.loc[public, "peak"].map(band)).sum().items()},
        "public_purchased": round(float(pub[b.loc[public, "source"] == "purchased"].sum())),
        "private_wells": round(float(pop[~public].sum())),
        "private_expected_over20": round(float((pop * b["p_over20"].fillna(0))[~public].sum())),
        "private_expected_detected": round(float((pop * b["p_detect"].fillna(0))[~public].sum())),
        "private_with_nearby_public_wells": round(float(pop[~public & (b["n_neigh"] > 0)].sum())),
    }


def block_group_table(b: pd.DataFrame) -> pd.DataFrame:
    b = b.assign(bg=b.index.str[:12])
    public = ~b["private_well"]
    pop = b["POP20"]
    w = pd.DataFrame({
        "bg": b["bg"], "pop": pop, "pub": pop * public, "prv": pop * ~public,
        "cur": (pop * b["current"]).where(public, 0.0), "pk": (pop * b["peak"]).where(public, 0.0),
        "p20": (pop * b["p_over20"]).where(~public, 0.0),
    }).fillna(0.0)
    g = w.groupby("bg").sum()
    first = b.groupby("bg")[["TOWN", "EJ", "EJ_CRIT_DE", "PCT_MINORI", "LIMENGHHPC", "BG_MHHI"]].first()
    systems = b[public].groupby("bg")["system"].agg(lambda s: "; ".join(sorted(set(s.dropna()))))
    out = pd.DataFrame({
        "town": first["TOWN"].str.title(), "population": g["pop"].round(),
        "share_public_water": (g["pub"] / g["pop"]).round(3),
        "water_systems": systems.reindex(g.index),
        "pfas6_current_ng_l": (g["cur"] / g["pub"]).replace([np.inf], np.nan).round(1),
        "pfas6_peak_ng_l": (g["pk"] / g["pub"]).replace([np.inf], np.nan).round(1),
        "share_private_wells": (g["prv"] / g["pop"]).round(3),
        "private_well_p_over20": (g["p20"] / g["prv"]).replace([np.inf], np.nan).round(3),
        "ej": first["EJ"].fillna("No"), "ej_criteria": first["EJ_CRIT_DE"],
        "pct_minority": first["PCT_MINORI"].round(1),
        "pct_limited_english_households": first["LIMENGHHPC"].round(1),
        "median_household_income": first["BG_MHHI"],
    })
    out.index.name = "block_group"
    return out


def system_table() -> pd.DataFrame:
    """Every community system with its PFAS6 levels and UCMR 5 cross-check."""
    sa = service_areas().drop(columns="geometry")
    t = sa.join(system_levels(sa["PWS_ID"]), on="PWS_ID").join(ucmr5_levels(), on="PWS_ID")
    t = t.rename(columns={"PWS_ID": "pws_id", "PWS_NAME": "system", "PWSPOP_WIN": "population_served",
                          "current": "pfas6_current_ng_l", "peak": "pfas6_peak_ng_l",
                          "source": "results_from", "via": "results_systems",
                          "ucmr5_pfas6": "ucmr5_pfas6_ng_l"})
    t["system"] = t["system"].str.title()
    return t.round({"pfas6_current_ng_l": 1, "pfas6_peak_ng_l": 1, "ucmr5_pfas6_ng_l": 1}).sort_values(
        "pfas6_current_ng_l", ascending=False)


def groundwater_validation(repeats: int = 5) -> dict:
    X, wells = training_table()
    out = {"wells": len(wells)}
    for name, threshold in THRESHOLDS.items():
        y = (wells["pfas6_max"].to_numpy() >= threshold).astype(int)
        m = evaluate(X, y, wells["TOWN"], repeats=repeats).groupby("model")[["roc_auc", "avg_precision"]].mean()
        out[name] = {"prevalence": round(float(y.mean()), 3),
                     **{k: {c: round(float(v), 3) for c, v in r.items()} for k, r in m.iterrows()}}
    return out


def exposure_report(blocks: pd.DataFrame, systems: pd.DataFrame) -> dict:
    from .water_map import private_band_counts
    both = systems.dropna(subset=["pfas6_current_ng_l", "ucmr5_pfas6_ng_l"])
    mwra = systems["results_systems"].fillna("").str.contains(MWRA_PWS)
    return {
        "latest_result": str(finished_results()["date"].max().date()),
        "mwra": {"systems": int(mwra.sum()), "population": int(systems.loc[mwra, "population_served"].sum()),
                 "pfas6_current_ng_l": float(systems.loc[systems["pws_id"] == MWRA_PWS, "pfas6_current_ng_l"].max())
                 if (systems["pws_id"] == MWRA_PWS).any() else 0.0},
        "headline": headline(blocks),
        "groups": summarize(blocks),
        "private_bands": private_band_counts(blocks),
        "groundwater_validation": groundwater_validation(),
        "ucmr5_check": {"systems": len(both), "correlation": round(float(
            both["pfas6_current_ng_l"].corr(both["ucmr5_pfas6_ng_l"])), 2)},
    }
