import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from pfas_risk import pfas_sources


def test_pws_results_use_only_samples_before_cutoff(tmp_path, monkeypatch):
    csv = tmp_path / "dw.csv"
    pd.DataFrame({
        "PWSId": ["1000001", "1000001", "1000001", "2000002"],
        "SampleLocCode": ["01G", "01G", "TP1", "01G"],
        "CollectedDate": ["2021-05-01T00:00:00", "2024-01-01T00:00:00", "2022-03-01T00:00:00",
                          "2023-06-01T00:00:00"],
        "Result": ["5.1", "80", "ND", "40"],
    }).to_csv(csv, index=False)
    sources = gpd.GeoDataFrame({"SOURCE_ID": ["1000001-01G", "1000001-02G", "2000002-01G"],
                                "PWS_ID": ["1000001", "1000001", "2000002"]},
                               geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], crs="EPSG:26986")
    monkeypatch.setattr(pfas_sources, "download", lambda name: csv)
    monkeypatch.setattr(pfas_sources, "read", lambda name: sources)
    r = pfas_sources.pws_pfas_results("2023-01-01").set_index("SOURCE_ID")["pfas6_max"]
    # 01G: its own 2021 sample (the 2024 one is after the cutoff); 02G: only the plant sample (ND);
    # system 2000002 was first sampled after the cutoff, so it has no result at all.
    assert r.to_dict() == {"1000001-01G": 5.1, "1000001-02G": 0.0}
