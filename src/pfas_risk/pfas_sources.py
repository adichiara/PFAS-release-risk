"""Specific PFAS source and evidence layers beyond the MassGIS and EPA facility data.

- Airports: FAA public-use aerodromes and military airfields (AFFF for crash rescue and training).
- Military installations: DoD MIRTA boundaries (AFFF, fire training areas).
- EPA Toxics Release Inventory reporters: facilities handling listed toxic chemicals. None in
  Massachusetts has reported a PFAS chemical, so this is a general industrial-chemicals layer.
- Public water supply sources with PFAS6 detected before a cutoff date. Drinking-water
  results often lead to a release notification, so only samples collected before the
  forward-test cutoff are used; the forward test then measures whether they anticipate
  releases reported later.
- Sewer service areas of municipal wastewater plants (septic systems and plant effluent
  and biosolids carry PFAS in different ways).
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .config import CRS, FORWARD_CUTOFF, WGS84
from .sources import download, read

PWS_DETECT_NG_L = 2.0     # the usual PFAS6 reporting limit
PWS_MCL_NG_L = 20.0       # Massachusetts PFAS6 drinking-water standard


def airport_points() -> gpd.GeoDataFrame:
    a = read("faa_airports")
    public = (a["TYPE_CODE"] == "AD") & (a["PRIVATEUSE"] == 0)
    military = a["MIL_CODE"].isin(["MIL", "ALL"])
    return a[public | military]


def military_areas() -> gpd.GeoDataFrame:
    return read("dod_mirta")


def tri_points() -> gpd.GeoDataFrame:
    t = pd.read_csv(download("epa_tri"), dtype=str,
                    usecols=["3. FRS ID", "4. FACILITY NAME", "12. LATITUDE", "13. LONGITUDE"])
    t.columns = ["frs_id", "name", "lat", "lon"]
    t[["lat", "lon"]] = t[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    t = t.dropna(subset=["lat", "lon"]).drop_duplicates("frs_id")
    return gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t["lon"], t["lat"]), crs=WGS84).to_crs(CRS)


def pws_pfas_results(cutoff: str = FORWARD_CUTOFF) -> pd.DataFrame:
    """Highest PFAS6 result (ng/L) per public water supply source, from samples before `cutoff`.

    A sample is tied to its source when its sampling-point code matches the source ID;
    samples taken at a treatment plant or entry point are tied to every source of the system.
    """
    d = pd.read_csv(download("massdep_dw_pfas"), dtype=str,
                    usecols=["PWSId", "SampleLocCode", "CollectedDate", "Result"])
    d = d[pd.to_datetime(d["CollectedDate"]) < pd.Timestamp(cutoff)]
    d["value"] = pd.to_numeric(d["Result"], errors="coerce").fillna(0.0)  # 'ND' -> 0
    d["source_id"] = d["PWSId"] + "-" + d["SampleLocCode"].str.strip()
    src = read("pws_sources")[["SOURCE_ID", "PWS_ID", "geometry"]]
    by_source = d[d["source_id"].isin(src["SOURCE_ID"])].groupby("source_id")["value"].max()
    by_system = d[~d["source_id"].isin(src["SOURCE_ID"])].groupby("PWSId")["value"].max()
    value = pd.concat([src["SOURCE_ID"].map(by_source), src["PWS_ID"].map(by_system)], axis=1).max(axis=1)
    return gpd.GeoDataFrame(src.assign(pfas6_max=value), geometry="geometry", crs=CRS).dropna(subset=["pfas6_max"])


def pws_pfas_points(cutoff: str = FORWARD_CUTOFF) -> dict[str, gpd.GeoDataFrame]:
    r = pws_pfas_results(cutoff)
    return {
        "pws_tested": r,
        "pws_pfas_detect": r[r["pfas6_max"] >= PWS_DETECT_NG_L],
        "pws_pfas_over20": r[r["pfas6_max"] >= PWS_MCL_NG_L],
    }


def sewer_areas() -> gpd.GeoDataFrame:
    return read("sewer_service")


def point_layers() -> dict[str, gpd.GeoDataFrame]:
    return {"airport": airport_points(), "tri": tri_points()} | pws_pfas_points()
