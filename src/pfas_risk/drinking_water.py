"""PFAS in the drinking water people actually receive, from measured results.

Every community water system in Massachusetts had to test finished (treated) water for PFAS6
under the 2020 state standard, so these results are a systematic measurement rather than a
record of where someone happened to look. For each community system:

- its own finished-water PFAS6 results (MassDEP, via the EEA Data Portal), or, if it buys its
  water and has none, the results of the systems it buys from (EPA SDWIS purchase links);
- ``current``: mean PFAS6 over the last 12 months of its results; ``peak``: highest annual
  mean in any calendar year (before treatment was added, for many systems).

Census blocks are assigned the community system whose service area they fall in. Blocks
outside every service area are presumed to be on private wells, which are not tested or
treated by a public system and are covered separately (groundwater model).

PFAS6 is the sum of PFOA, PFOS, PFHxS, PFNA, PFHpA and PFDA; the Massachusetts standard is
20 ng/L. Non-detects count as zero.
"""

from __future__ import annotations

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS
from .sources import download, read

MCL_NG_L = 20.0
# Bands for PFAS6 in ng/L: not detected, low, elevated, at or above the state standard.
BANDS = [(-np.inf, 2.0, "Not detected (<2)"), (2.0, 10.0, "2 to <10"), (10.0, 20.0, "10 to <20"),
         (20.0, np.inf, "20 or more (state standard)")]
PFAS6 = ["PFOA", "PFOS", "PFHxS", "PFNA", "PFHpA", "PFDA"]


def band(v: float) -> str | None:
    if pd.isna(v):
        return None
    return next(label for lo, hi, label in BANDS if lo <= v < hi)


def finished_results() -> pd.DataFrame:
    """Finished-water PFAS6 results per system: PWSId, date, value (ng/L; ND = 0)."""
    d = pd.read_csv(download("massdep_dw_pfas"), dtype=str,
                    usecols=["PWSId", "RaworFinished", "CollectedDate", "Result"])
    d = d[d["RaworFinished"] == "F"]
    return pd.DataFrame({"pws": d["PWSId"].str.zfill(7),
                         "date": pd.to_datetime(d["CollectedDate"]),
                         "value": pd.to_numeric(d["Result"], errors="coerce").fillna(0.0)})


def system_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per system: current (last 12 months of its data) and peak annual mean, sample count."""
    last = results.groupby("pws")["date"].transform("max")
    recent = results[results["date"] > last - pd.DateOffset(months=12)]
    annual = results.groupby(["pws", results["date"].dt.year])["value"].mean()
    return pd.DataFrame({
        "current": recent.groupby("pws")["value"].mean(),
        "peak": annual.groupby(level=0).max(),
        "samples": results.groupby("pws").size(),
        "last_sample": results.groupby("pws")["date"].max(),
    })


def purchases() -> pd.DataFrame:
    """Active purchased-water links (buyer -> seller) from EPA SDWIS, as 7-digit state IDs."""
    s = pd.read_csv(download("sdwis_sellers"), dtype=str)
    s = s[(s["facility_activity_code"] == "A") & s["availability_code"].isin(["P", "S"])]
    return pd.DataFrame({"buyer": s["pwsid"].str[2:], "seller": s["seller_pwsid"].str[2:]}).drop_duplicates()


def system_levels(systems: pd.Series) -> pd.DataFrame:
    """PFAS6 levels for each community system, falling back to its sellers' results.

    A system with its own finished-water results uses them (its sampling points include any
    blended purchased water). A system without them takes the mean of its sellers' levels,
    following purchase links up to three steps (e.g. a town buying from a town buying from MWRA).
    """
    own = system_summary(finished_results())
    links = purchases()
    rows = {}
    for pws in systems.unique():
        frontier, source, seen = [pws], "own", set()
        for _ in range(4):
            found = own.reindex([p for p in frontier if p in own.index])
            if len(found):
                rows[pws] = {"current": found["current"].mean(), "peak": found["peak"].mean(),
                             "source": source, "via": ",".join(found.index)}
                break
            seen.update(frontier)
            frontier = [s for s in links.loc[links["buyer"].isin(frontier), "seller"] if s not in seen]
            source = "purchased"
            if not frontier:
                break
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.reindex(systems.unique())


def ucmr5_levels() -> pd.DataFrame:
    """EPA UCMR 5 (2023-2025): mean over samples of the six PFAS6 compounds summed, ng/L."""
    with zipfile.ZipFile(download("ucmr5")) as z:
        u = pd.read_csv(z.open("UCMR5_All.txt"), sep="\t", dtype=str, encoding="latin1",
                        usecols=["PWSID", "State", "SampleID", "Contaminant", "AnalyticalResultValue"])
    u = u[(u["State"] == "MA") & u["Contaminant"].isin(PFAS6)]
    u["value"] = pd.to_numeric(u["AnalyticalResultValue"], errors="coerce").fillna(0.0) * 1000  # µg/L -> ng/L
    per_sample = u.groupby(["PWSID", "SampleID"])["value"].sum()
    out = per_sample.groupby(level=0).agg(["mean", "size"]).rename(columns={"mean": "ucmr5_pfas6", "size": "ucmr5_samples"})
    out.index = out.index.str[2:]
    return out


def service_areas() -> gpd.GeoDataFrame:
    sa = read("water_service_areas")[["PWS_ID", "PWS_NAME", "PWSPOP_WIN", "geometry"]]
    sa["PWS_ID"] = sa["PWS_ID"].astype(str).str.zfill(7)
    return sa


def block_water(blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each block: its community system (or none: private wells) and that system's PFAS6."""
    sa = service_areas()
    rep = gpd.GeoDataFrame(geometry=blocks.representative_point(), crs=CRS)
    j = gpd.sjoin(rep, sa, predicate="within", how="left")
    # Overlapping service areas: keep the larger system (usually the main municipal supplier).
    j = j.sort_values("PWSPOP_WIN", ascending=False)
    j = j[~j.index.duplicated()]
    levels = system_levels(sa["PWS_ID"])
    out = pd.DataFrame({"pws": j["PWS_ID"], "system": j["PWS_NAME"]}, index=j.index)
    out = out.join(levels, on="pws")
    out["private_well"] = out["pws"].isna()
    return out.reindex(blocks.index)
