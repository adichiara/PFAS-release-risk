import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from pfas_risk import drinking_water, groundwater
from pfas_risk.drinking_water import band, system_levels, system_summary


def test_bands_follow_the_state_standard():
    assert band(0) == "Not detected (<2)"
    assert band(19.9) == "10 to <20"
    assert band(20) == "20 or more (state standard)"
    assert band(np.nan) is None


def test_system_summary_uses_last_12_months_and_worst_year():
    r = pd.DataFrame({"pws": ["1"] * 4, "date": pd.to_datetime(["2021-03-01", "2021-09-01", "2025-01-01", "2025-06-01"]),
                      "value": [30.0, 20.0, 4.0, 2.0]})
    s = system_summary(r).loc["1"]
    assert s["current"] == 3.0 and s["peak"] == 25.0


def test_purchased_water_takes_the_sellers_levels(monkeypatch):
    results = pd.DataFrame({"pws": ["6000000", "2000000"], "date": pd.to_datetime(["2025-01-01"] * 2),
                            "value": [0.0, 12.0]})
    links = pd.DataFrame({"buyer": ["3035000", "3999000"], "seller": ["6000000", "3035000"]})
    monkeypatch.setattr(drinking_water, "finished_results", lambda: results)
    monkeypatch.setattr(drinking_water, "purchases", lambda: links)
    lv = system_levels(pd.Series(["2000000", "3035000", "3999000", "4444444"]))
    assert lv.loc["2000000", "source"] == "own" and lv.loc["2000000", "current"] == 12.0
    assert lv.loc["3035000", "source"] == "purchased" and lv.loc["3035000", "current"] == 0.0
    assert lv.loc["3999000", "via"] == "6000000"          # two purchase steps
    assert pd.isna(lv.loc["4444444", "current"])


def test_neighbour_evidence_skips_the_wells_own_system():
    wells = gpd.GeoDataFrame({"PWS_ID": ["A", "A", "B"], "pfas6_max": [100.0, 100.0, 0.0]},
                             geometry=[Point(0, 0), Point(300, 0), Point(600, 0)], crs="EPSG:26986")
    ev = groundwater.neighbour_evidence(wells.geometry, wells, own_pws=wells["PWS_ID"].to_numpy())
    assert ev.loc[0, "n_neigh"] == 1 and ev.loc[0, "neigh_log_pfas6"] == 0.0     # only system B counts
    assert ev.loc[2, "n_neigh"] == 2 and ev.loc[2, "neigh_log_pfas6"] > 4


def test_raw_samples_match_their_source(tmp_path, monkeypatch):
    csv = tmp_path / "dw.csv"
    pd.DataFrame({"PWSId": ["2002000", "2002000", "3000001"], "RaworFinished": ["R", "R", "R"],
                  "SampleLocCode": ["RW-03G", "03G-RW", "10001"], "Result": ["25", "ND", "7"]}).to_csv(csv, index=False)
    src = gpd.GeoDataFrame({"SOURCE_ID": ["2002000-03G", "2002000-04G", "3000001-01G"],
                            "PWS_ID": ["2002000", "2002000", "3000001"], "TOWN": ["ACTON"] * 3,
                            "TYPE": ["GW"] * 3}, geometry=[Point(0, 0)] * 3, crs="EPSG:26986")
    monkeypatch.setattr(groundwater, "download", lambda name: csv)
    monkeypatch.setattr(groundwater, "read", lambda name: src)
    w = groundwater.well_results()["pfas6_max"].to_dict()
    # 03G matched by code; 3000001 has a single groundwater source, so its sample belongs to it.
    assert w == {"2002000-03G": 25.0, "3000001-01G": 7.0}
